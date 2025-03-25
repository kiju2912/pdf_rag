import os
import re
import random
import fitz  # PyMuPDF: PDF 파일 읽기 및 조작 라이브러리
import math
from collections import deque
import mysql.connector
from datetime import datetime
import time
from logic.image_caption import get_image_captions


# 그룹화 허용 오차: 텍스트 블록들의 x 좌표 차이 허용치 (컬럼 검출에 사용)
GROUP_TOLERANCE = 10

def rect_overlap_ratio(r1, r2):
    """
    두 사각형(r1, r2) 간의 겹치는 면적 비율을 계산합니다.
    - 겹치는 영역의 면적을 두 사각형 중 작은 면적에 대해 비율로 반환합니다.
    - 캡션이나 영역의 중복 여부 판단 등 영역 비교에 사용됨.
    """
    x0 = max(r1.x0, r2.x0)
    y0 = max(r1.y0, r2.y0)
    x1 = min(r1.x1, r2.x1)
    y1 = min(r1.y1, r2.y1)
    if x1 > x0 and y1 > y0:
        inter_area = (x1 - x0) * (y1 - y0)
        return inter_area / min(r1.get_area(), r2.get_area())
    return 0

def already_drawn(page_number, rect, drawn_list, threshold=0.8):
    """
    이미 그려진(rectangles) 영역과 현재 rect의 겹침 비율이 임계값(threshold) 이상이면
    이미 처리된 영역으로 판단합니다.
    주로 중복으로 캡션이나 클러스터 영역을 표시하지 않도록 하기 위해 사용됨.
    """
    for pn, r in drawn_list:
        if page_number == pn and rect_overlap_ratio(rect, r) > threshold:
            return True
    return False

def is_in_blocks(page_number, rect, drawn_list):
    """
    drawn_list에 있는 사각형 중 하나라도 현재 rect를 포함하거나 교차하는지 검사합니다.
    이미 처리된 영역과의 중복을 방지하는 역할.
    """
    for tup in drawn_list:
        pn, r = tup[0], tup[1]
        if page_number == pn and (r.contains(rect) or r.intersects(rect)):
            return True
    return False

def is_intersects_blocks(page_number, rect, drawn_list):
    """
    drawn_list에 있는 사각형 중 하나라도 현재 rect와 교차하는지 확인합니다.
    영역 간의 겹침 여부를 보다 단순하게 체크할 때 사용됨.
    """
    for tup in drawn_list:
        pn, r = tup[0], tup[1]
        if page_number == pn and r.intersects(rect):
            return True
    return False

def is_near(r1, r2, threshold):
    """
    두 사각형(r1, r2)이 threshold 이내에 위치하는지 판단합니다.
    - 두 사각형 사이의 최소 거리(두 사각형 경계 사이의 거리)를 계산하고,
    그 값이 threshold 이하이면 가까이 있다고 판단.
    - 클러스터링 알고리즘에서 인접 요소 그룹화에 사용됨.
    """
    dx = max(r1.x0 - r2.x1, r2.x0 - r1.x1, 0)
    dy = max(r1.y0 - r2.y1, r2.y0 - r1.y1, 0)
    return (dx**2 + dy**2)**0.5 <= threshold

def cluster_elements(rects, threshold=5):
    """
    비텍스트 요소(이미지, 드로잉 등)들의 사각형 리스트를 threshold 값을 기준으로 클러스터링합니다.
    - 너비, 높이 등의 거리를 비교하여 인접한 요소들을 하나의 그룹(클러스터)으로 묶습니다.
    - BFS/DFS와 유사하게 queue를 사용하여 인접한 모든 요소를 방문합니다.
    """
    clusters = []
    visited = set()
    for i, rect in enumerate(rects):
        if i in visited:
            continue
        cluster = []
        queue = deque([i])
        while queue:
            idx = queue.popleft()
            if idx in visited:
                continue
            visited.add(idx)
            cluster.append(rects[idx])
            for j, other in enumerate(rects):
                if j not in visited and is_near(rects[idx], other, threshold):
                    queue.append(j)
        clusters.append(cluster)
    return clusters

def merge_overlapping_rects(rects, tol=0):
    """
    입력된 사각형 리스트 중 서로 겹치거나 인접(tol 이하 차이)하는 사각형들을 반복적으로 합칩니다.
    - PDF 영역 추출 후 중복 또는 분할된 영역을 하나로 병합하는 알고리즘.
    - while 루프를 사용하여 반복적으로 병합이 일어날 때까지 수행.
    """
    if not rects:
        return []
    merged = rects.copy()
    changed = True
    while changed:
        changed = False
        new_merged = []
        while merged:
            current = merged.pop(0)
            i = 0
            while i < len(merged):
                if current.intersects(merged[i]) or current.contains(merged[i]) or merged[i].contains(current):
                    current |= merged.pop(i)  # 두 사각형의 합집합을 계산
                    changed = True
                else:
                    i += 1
            new_merged.append(current)
        merged = new_merged
    return merged

def subtract_rect(original, subtract):
    """
    original 사각형에서 subtract 사각형과의 교집합 영역을 제거한 나머지 후보 영역들을 반환합니다.
    - 상, 하, 좌, 우 영역을 각각 후보로 추출합니다.
    - 캡션 영역 재탐지나 클러스터링 전 후보 영역 분리 시 사용됨.
    """
    if not original.intersects(subtract):
        return [original]
    inter = original & subtract  # 두 사각형의 교집합
    candidates = []
    if inter.y0 > original.y0:
        candidates.append(fitz.Rect(original.x0, original.y0, original.x1, inter.y0))
    if inter.y1 < original.y1:
        candidates.append(fitz.Rect(original.x0, inter.y1, original.x1, original.y1))
    if inter.x0 > original.x0:
        candidates.append(fitz.Rect(original.x0, inter.y0, inter.x0, inter.y1))
    if inter.x1 < original.x1:
        candidates.append(fitz.Rect(inter.x1, inter.y0, original.x1, inter.y1))
    return [c for c in candidates if c.get_area() > 0]

def closest_points_between_rectangles(r1, r2):
    """
    두 사각형(r1, r2) 경계 상에서 서로의 최소 거리를 이루는 두 점을 반환합니다.
    - 두 사각형의 상대적인 위치(겹치는지, 떨어져 있는지)에 따라
      서로 다른 점을 선택하여 최소 거리를 계산합니다.
    - 캡션과 클러스터 영역 매칭 시 두 영역 간의 거리를 계산하는 데 사용됨.
    """
    if r1.x1 < r2.x0:
        p1_x = r1.x1
        p2_x = r2.x0
    elif r2.x1 < r1.x0:
        p1_x = r1.x0
        p2_x = r2.x1
    else:
        overlap_x0 = max(r1.x0, r2.x0)
        overlap_x1 = min(r1.x1, r2.x1)
        p1_x = p2_x = (overlap_x0 + overlap_x1) / 2

    if r1.y1 < r2.y0:
        p1_y = r1.y1
        p2_y = r2.y0
    elif r2.y1 < r1.y0:
        p1_y = r1.y0
        p2_y = r2.y1
    else:
        overlap_y0 = max(r1.y0, r2.y0)
        overlap_y1 = min(r1.y1, r2.y1)
        p1_y = p2_y = (overlap_y0 + overlap_y1) / 2

    return (p1_x, p1_y), (p2_x, p2_y)

def is_in_matched(cluster, matched_list):
    """
    클러스터(사각형)와 이미 매칭된 사각형들(matched_list) 중
    거의 동일한 위치에 존재하는지 (좌표 차이가 매우 작음) 확인합니다.
    - 매칭 후 후처리 단계에서 중복 클러스터 병합 여부 판단에 사용.
    """
    for _, m in matched_list:
        if (abs(cluster.x0 - m.x0) < 1e-3 and abs(cluster.y0 - m.y0) < 1e-3 and
            abs(cluster.x1 - m.x1) < 1e-3 and abs(cluster.y1 - m.y1) < 1e-3):
            return True
    return False

def save_to_sql(file_name, table_caption_regions, figure_caption_regions, drawn_table_regions, page_caption_matching):
    # MySQL 연결 정보 수정 (호스트, 사용자, 패스워드, 데이터베이스 이름)
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='00000000',
        database='pdf_parser',
        port=3306
    )

    cursor = conn.cursor()
    
    # 1. pdf_documents 테이블에 파일 정보 저장
    cursor.execute("INSERT INTO pdf_documents (file_name) VALUES (%s)", (file_name,))
    pdf_id = cursor.lastrowid
    
    # 캡션 정보를 저장하기 위한 매핑 (캡션 라벨 -> caption_id)
    caption_mapping = {}
    
    # 2. 캡션 정보를 저장 (table_caption_regions, figure_caption_regions 모두)
    # 각 튜플은 (cap_rect, cap_label, cap_text, page_number, [pdf_file_name], [png_file_name]) 형식이어야 함
    for region in table_caption_regions + figure_caption_regions:
        cap_rect, cap_label, cap_text, page_number = region[:4]
        pdf_file_name = region[4] if len(region) >= 5 else ""
        # png_file_name이 존재하면 가져오고, 없으면 빈 문자열
        png_file_name = region[5] if len(region) >= 6 else ""
        x0, y0, x1, y1 = cap_rect.x0, cap_rect.y0, cap_rect.x1, cap_rect.y1
        cursor.execute(
            "INSERT INTO captions (caption_name, pdf_id, page_number, caption_text, x0, y0, x1, y1) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (cap_label, pdf_id, page_number, cap_text, x0, y0, x1, y1)
        )
        caption_id = cursor.lastrowid
        caption_mapping[cap_label] = caption_id

    # 매핑: table 캡션 라벨 -> pdf 파일 경로 및 png 파일 경로 (table_caption_regions에 해당 값들이 추가됨)
    table_pdf_mapping = {cap_label: pdf_file_name for (_, cap_label, _, _, pdf_file_name, *_) in table_caption_regions}
    table_png_mapping = {cap_label: png_file_name for (_, cap_label, _, _, _, png_file_name) in table_caption_regions}
    
    # 3. 그려진 표 영역 정보를 저장 (drawn_table_regions) → area 테이블에 type 'table'로 저장
    for page_number, table_rect, cap_label in drawn_table_regions:
        caption_id = caption_mapping.get(cap_label)
        if caption_id is None:
            continue
        x0, y0, x1, y1 = table_rect.x0, table_rect.y0, table_rect.x1, table_rect.y1
        pdf_file_name = table_pdf_mapping.get(cap_label, '')
        png_file_name = table_png_mapping.get(cap_label, '')
        
        appearance_description = get_image_captions(png_file_name)
        cursor.execute(
            "INSERT INTO area (caption_id, pdf_file_name, png_file_name, page_number, x0, y0, x1, y1, type, appearance_description) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (caption_id, pdf_file_name, png_file_name, page_number, x0, y0, x1, y1, 'table', appearance_description)
        )
    
    # 4. 클러스터 영역 정보를 저장 (page_caption_matching) → area 테이블에 type 'figure'로 저장
    for page_number, matching in page_caption_matching.items():
        for cap_label, match_data in matching.items():
            # match_data: (cluster_rect, p_cluster, p_cap, distance, pdf_file_name, [png_file_name])
            cluster_rect, p_cluster, p_cap, distance = match_data[:4]
            pdf_file_name = match_data[4] if len(match_data) >= 5 else ""
            png_file_name = match_data[5] if len(match_data) >= 6 else ""
            caption_id = caption_mapping.get(cap_label)
            appearance_description = get_image_captions(png_file_name)
            if caption_id is None:
                continue
            x0, y0, x1, y1 = cluster_rect.x0, cluster_rect.y0, cluster_rect.x1, cluster_rect.y1
            cursor.execute(
                "INSERT INTO area (caption_id, pdf_file_name, png_file_name, page_number, x0, y0, x1, y1, type, appearance_description) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (caption_id, pdf_file_name, png_file_name, page_number, x0, y0, x1, y1, 'figure', appearance_description)
            )
    
    conn.commit()
    cursor.close()
    conn.close()
    return pdf_id

def save_page_as_png(page, cap_label, timestamp, png_folder):
    """
    주어진 페이지 영역을 PNG 파일로 저장합니다.
    :param page: fitz.Page 객체
    :param cap_label: 캡션 라벨 (파일 이름 생성에 사용)
    :param timestamp: 시간 스탬프 (파일 이름 생성에 사용)
    :param png_folder: 저장될 PNG 폴더 경로 (예: output/png/문서명)
    :return: PNG 파일 경로
    """
    os.makedirs(png_folder, exist_ok=True)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    png_filename = f"{cap_label}_{timestamp}.png"
    png_filepath = os.path.join(png_folder, png_filename)
    pix.save(png_filepath)
    return png_filepath


def save_regions_as_pdf(doc, regions, document_name, drawn_table_regions):
    """
    regions: 리스트 (cap_rect, cap_label, cap_text, page_number)
    저장 후 각 튜플에 pdf_file_name과 png_file_name을 추가하여 리턴
    경로: PDF는 output/pdf/문서명/파일명, PNG는 output/png/문서명/파일명
    """
    pdf_folder = os.path.join("output", "pdf", document_name)
    png_folder = os.path.join("output", "png", document_name)
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)
    
    updated_regions = []
    for region in regions:
        cap_rect, cap_label, cap_text, page_number = region[:4]
        # drawn_table_regions에서 동일한 페이지와 캡션 라벨에 해당하는 테이블 영역 찾기
        table_region = None
        for entry in drawn_table_regions:
            pn, table_rect, table_cap_label = entry
            if pn == page_number and table_cap_label == cap_label:
                table_region = table_rect
                break
        clip_rect = table_region if table_region is not None else cap_rect
        
        new_doc = fitz.open()  # 빈 PDF 문서 생성
        new_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)
        new_page = new_doc[0]
        # 페이지(MediaBox)와의 교집합 계산
        clip_rect = clip_rect & new_page.rect
        if clip_rect.is_empty or not new_page.rect.contains(clip_rect):
            clip_rect = new_page.rect
        new_page.set_cropbox(clip_rect)
        
        timestamp = time.time_ns()
        pdf_filename = f"{cap_label}_{timestamp}.pdf"
        pdf_filepath = os.path.join(pdf_folder, pdf_filename)
        new_doc.save(pdf_filepath)
        
        # 별도 함수 호출하여 PNG로 저장
        png_filepath = save_page_as_png(new_page, cap_label, timestamp, png_folder)
        
        new_doc.close()
        updated_regions.append((cap_rect, cap_label, cap_text, page_number, pdf_filepath, png_filepath))
    return updated_regions


def save_cluster_regions_as_pdf(doc, page_caption_matching, document_name):
    """
    page_caption_matching: {page_number: {cap_label: (cluster_rect, p_cluster, p_cap, distance)}}
    저장 후 각 매칭 튜플에 pdf_file_name과 png_file_name을 추가하여 리턴
    경로: PDF는 output/pdf/문서명/파일명, PNG는 output/png/문서명/파일명
    """
    pdf_folder = os.path.join("output", "pdf", document_name)
    png_folder = os.path.join("output", "png", document_name)
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(png_folder, exist_ok=True)
    
    updated_page_caption_matching = {}
    for page_number, captions in page_caption_matching.items():
        updated_captions = {}
        for cap_label, match_data in captions.items():
            # match_data: (cluster_rect, p_cluster, p_cap, distance)
            cluster_rect, p_cluster, p_cap, distance = match_data[:4]
            timestamp = time.time_ns()
            pdf_filename = f"{cap_label}_{timestamp}.pdf"
            pdf_filepath = os.path.join(pdf_folder, pdf_filename)
            
            new_doc = fitz.open()
            new_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)
            new_page = new_doc[0]
            # 페이지(MediaBox)와의 교집합 계산
            cluster_rect = cluster_rect & new_page.rect
            if cluster_rect.is_empty or not new_page.rect.contains(cluster_rect):
                cluster_rect = new_page.rect
            new_page.set_cropbox(cluster_rect)
            new_doc.save(pdf_filepath)
            
            # 별도 함수 호출하여 PNG로 저장
            png_filepath = save_page_as_png(new_page, cap_label, timestamp, png_folder)
            
            new_doc.close()
            updated_captions[cap_label] = (cluster_rect, p_cluster, p_cap, distance, pdf_filepath, png_filepath)
        updated_page_caption_matching[page_number] = updated_captions
    return updated_page_caption_matching



def process_pdf(input_path, output_path):
    """
    전체 PDF 처리 파이프라인:
      1. PDF 파일 열기 및 텍스트, 이미지, 드로잉 요소 추출
      2. 캡션(테이블, 피규어) 검출 및 영역 추출
      3. 텍스트 블록을 기반으로 페이지 내 컬럼(열) 검출
      4. 캡션과 관련된 가로선(라인) 분석을 통해 테이블 영역 결정
      5. 이미지 및 드로잉 요소를 클러스터링하여 비텍스트 영역 검출
      6. 클러스터 영역 내 캡션 재탐지 및 텍스트 보강 클러스터링
      7. 캡션과 클러스터 영역을 DFS 기반 매칭 알고리즘을 통해 1:1 매칭 수행
      8. 매칭되지 않은 클러스터 영역에 대해 후처리(병합) 수행
      9. 캡션/클러스터 영역을 개별 PDF로 저장하고, 최종 결과를 PDF에 시각화
      10. 최종 정보를 SQL 데이터베이스에 저장
    """
    doc = fitz.open(input_path)
    global_main_blocks = []  # (페이지 번호, 사각형, 텍스트)
    table_caption_regions = []  # (캡션 사각형, 라벨, 캡션 텍스트, 페이지 번호)
    figure_caption_regions = []  # (캡션 사각형, 라벨, 캡션 텍스트, 페이지 번호)
    drawn_table_regions = []  # (페이지 번호, 테이블 영역 사각형, 캡션 라벨)
    drawn_rectangles = []     # (페이지 번호, 사각형) - 이미 처리된 영역 저장

    # 캡션과 클러스터 매칭 정보를 저장할 딕셔너리
    page_caption_matching = {}
    # 페이지 내 전체 컬럼 영역: 추후 이미지/드로잉 요소의 제외 범위 결정에 사용
    entire_col_rect = fitz.Rect()

    # ── 그리기 작업(pending drawing instructions) 저장 리스트 ──
    pending_rect_draws = []          # (페이지 번호, 사각형, 색상, 선 두께)
    pending_main_text_rect_draws = []  # (페이지 번호, 사각형, 색상, 선 두께)
    pending_text_inserts = []        # (페이지 번호, 텍스트, 위치, 색상, 폰트 크기)
    pending_line_draws = []          # (페이지 번호, 시작점, 종료점, 색상, 선 두께)

    # 캡션 판별용 정규 표현식: 피규어와 테이블
    fig_pattern = re.compile(r'(?i)^(fig(?:ure)?\.?|첨부자료|첨부파일)(\d+(?:\.\d+)?)(?P<special>.)')
    table_pattern = re.compile(r'(?i)^(table|테이블)(\d+(?:\.\d+)?)(?P<special>.)')

    # ── 1. 텍스트 블록 처리: 각 페이지별 텍스트 블록 추출 및 캡션 검출 ──
    for page in doc:
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            text = block[4].strip()
            if not text:
                continue
            text_no_space = re.sub(r'\s+', '', text)
            match_fig = fig_pattern.match(text_no_space)
            match_table = table_pattern.match(text_no_space)
            if match_fig:
                # 피규어 캡션 검출: 특수문자가 포함된 경우에만 캡션으로 판단
                special_char = match_fig.group("special")
                if special_char.isalnum() or special_char.isspace():
                    continue
                cap_rect = fitz.Rect(block[:4])
                if cap_rect.get_area() <= 0:
                    continue
                fig_label = f"Figure {match_fig.group(2)}"
                pending_rect_draws.append((page.number, cap_rect, (1, 0, 0), 2))
                pending_text_inserts.append((page.number, fig_label, (cap_rect.x0, cap_rect.y0), (0, 0, 0), 12))
                drawn_rectangles.append((page.number, cap_rect))
                figure_caption_regions.append((cap_rect, fig_label, text, page.number))
            elif match_table:
                # 테이블 캡션 검출: 피규어와 동일한 로직 사용
                special_char = match_table.group("special")
                if special_char.isalnum() or special_char.isspace():
                    continue
                cap_rect = fitz.Rect(block[:4])
                if cap_rect.get_area() <= 0:
                    continue
                table_label = f"Table {match_table.group(2)}"
                pending_rect_draws.append((page.number, cap_rect, (1, 0, 0), 2))
                pending_text_inserts.append((page.number, table_label, (cap_rect.x0, cap_rect.y0), (0, 0, 0), 12))
                table_caption_regions.append((cap_rect, table_label, text, page.number))
                drawn_rectangles.append((page.number, cap_rect))
            else:
                block_rect = fitz.Rect(block[:4])
                global_main_blocks.append((page.number, block_rect, text))
    
    # ── 2. 열(컬럼) 검출: 본문 텍스트 블록 기반으로 페이지 내 열 영역 추출 ──
    for page in doc:
        page_main_blocks = [entry for entry in global_main_blocks if entry[0] == page.number]
        page_columns = []
        remaining_blocks = page_main_blocks.copy()
        while remaining_blocks:
            groups = []
            for entry in remaining_blocks:
                _, rect, _ = entry
                placed = False
                for group in groups:
                    _, rep_rect, _ = group[0]
                    if abs(rect.x0 - rep_rect.x0) <= GROUP_TOLERANCE and abs(rect.x1 - rep_rect.x1) <= GROUP_TOLERANCE:
                        group.append(entry)
                        placed = True
                        break
                if not placed:
                    groups.append([entry])
            dominant = max(groups, key=lambda g: sum(entry[1].get_area() for entry in g))
            col_x_min = min(entry[1].x0 for entry in dominant)
            col_x_max = max(entry[1].x1 for entry in dominant)
            col_y_min = min(entry[1].y0 for entry in dominant)
            col_y_max = max(entry[1].y1 for entry in dominant)
            page_columns.append((col_x_min, col_x_max, col_y_min, col_y_max))
            remaining_blocks = [entry for entry in remaining_blocks if not (entry[1].x1 > col_x_min and entry[1].x0 < col_x_max)]
    
        # 페이지 내 가로선 후보 추출: 높이 < 2, 너비 > 20인 선분
        horz_lines = []
        for obj in page.get_drawings():
            if "rect" not in obj:
                continue
            r = obj["rect"]
            if (r.y1 - r.y0) < 2 and (r.x1 - r.x0) > 20:
                horz_lines.append(r)
    
        tol_x = 10  # 캡션 중앙 기준 x 오차 허용
    
        # ── 3. 테이블 캡션 블럭 처리: 캡션의 위치와 가로선 정보로 테이블 영역 결정 ──
        for cap_rect, cap_label, text, cap_page in table_caption_regions:
            if cap_page != page.number:
                continue
            cap_center_x = (cap_rect.x0 + cap_rect.x1) / 2
            cap_center_y = (cap_rect.y0 + cap_rect.y1) / 2
    
            selected_lines = []
            for line in horz_lines:
                line_center_x = (line.x0 + line.x1) / 2
                if abs(line_center_x - cap_center_x) <= tol_x:
                    selected_lines.append(line)
            is_iside = False
            if not selected_lines:
                col_range = None
                for col in page_columns:
                    col_x_min, col_x_max, _, _ = col
                    if cap_center_x >= col_x_min and cap_center_x <= col_x_max:
                        col_range = (col_x_min, col_x_max)
                        break
                if col_range is None:
                    col_range = (cap_rect.x0, cap_rect.x1)
                for line in horz_lines:
                    is_iside = True
                    if line.x0 <= col_range[1] + tol_x and line.x1 >= col_range[0] - tol_x:
                        if line not in selected_lines:
                            selected_lines.append(line)
            if not selected_lines:
                table_rect = fitz.Rect(col_range[0], cap_rect.y1, col_range[1], cap_rect.y1 + 20)
                if any(pn == page.number and rect_overlap_ratio(r, table_rect) > 0.8 for (pn, r, _) in drawn_table_regions):
                    pending_rect_draws.append((page.number, table_rect, (0, 1, 0), 2))
                    pending_text_inserts.append((page.number, cap_label, (table_rect.x0, table_rect.y0 - 10), (0, 0, 1), 12))
                else:
                    pending_rect_draws.append((page.number, table_rect, (0, 1, 0), 2))
                    pending_text_inserts.append((page.number, cap_label, (table_rect.x0, table_rect.y0 - 10), (0, 1, 0), 12))
                    drawn_table_regions.append((page.number, table_rect, cap_label))
                continue
    
            best_group = selected_lines
            closest_line = min(best_group, key=lambda r: abs(((r.y0 + r.y1) / 2) - cap_center_y))
            closest_line_center_y = (closest_line.y0 + closest_line.y1) / 2
            direction = 1 if closest_line_center_y - cap_center_y > 0 else -1
            candidate_lines = []
            for line in selected_lines:
                line_center_y = (line.y0 + line.y1) / 2
                if ((line_center_y - cap_center_y) * (closest_line_center_y - cap_center_y) > 0 and not line.intersects(cap_rect)) or is_iside:
                    candidate_lines.append(line)
            boundary_y = None
            for other_cap_rect, other_cap_label, other_text, other_cap_page in table_caption_regions:
                if other_cap_page != page.number:
                    continue
                other_center_x = (other_cap_rect.x0 + other_cap_rect.x1) / 2
                other_center_y = (other_cap_rect.y0 + other_cap_rect.y1) / 2
                if other_cap_rect == cap_rect or abs(other_center_x - cap_center_x) > tol_x:
                    continue
                if direction == 1 and other_center_y > cap_center_y:
                    if boundary_y is None or other_center_y < boundary_y:
                        boundary_y = other_center_y
                elif direction == -1 and other_center_y < cap_center_y:
                    if boundary_y is None or other_center_y > boundary_y:
                        boundary_y = other_center_y
            filtered_candidates = []
            if not is_iside:
                for line in candidate_lines:
                    line_center_y = (line.y0 + line.y1) / 2
                    if direction == 1:
                        if line_center_y > cap_center_y and (boundary_y is None or line_center_y < boundary_y):
                            filtered_candidates.append(line)
                    else:
                        if line_center_y < cap_center_y and (boundary_y is None or line_center_y > boundary_y):
                            filtered_candidates.append(line)
            else:
                filtered_candidates = candidate_lines
    
            x_tol2 = 10
            x_groups = []
            for line in filtered_candidates:
                placed = False
                for group in x_groups:
                    rep_line = group[0]
                    if abs(line.x0 - rep_line.x0) <= x_tol2 and abs(line.x1 - rep_line.x1) <= x_tol2:
                        group.append(line)
                        placed = True
                        break
                if not placed:
                    x_groups.append([line])
    
            closest_group = None
            for group in x_groups:
                if closest_line in group:
                    closest_group = group
                    break
            if closest_group:
                refined_x_min = min(r.x0 for r in closest_group)
                refined_x_max = max(r.x1 for r in closest_group)
                refined_y_min = min(r.y0 for r in closest_group)
                refined_y_max = max(r.y1 for r in closest_group)
                refined_rect = fitz.Rect(refined_x_min, refined_y_min, refined_x_max, refined_y_max)
                if any(pn == page.number and rect_overlap_ratio(r, refined_rect) > 0.8 for (pn, r, _) in drawn_table_regions):
                    pending_rect_draws.append((page.number, refined_rect, (0, 1, 0), 2))
                    pending_text_inserts.append((page.number, cap_label, (refined_rect.x0, refined_rect.y0 - 10), (0, 1, 0), 12))
                    drawn_rectangles.append((page.number, refined_rect))
                    new_candidates = [line for line in selected_lines if line not in closest_group]
                    if new_candidates:
                        x_groups_new = []
                        for line in new_candidates:
                            placed = False
                            for group in x_groups_new:
                                rep_line = group[0]
                                if abs(line.x0 - rep_line.x0) <= x_tol2 and abs(line.x1 - rep_line.x1) <= x_tol2:
                                    group.append(line)
                                    placed = True
                                    break
                            if not placed:
                                x_groups_new.append([line])
                        new_closest_group = None
                        new_closest_line = None
                        min_diff = float('inf')
                        for group in x_groups_new:
                            for line in group:
                                line_center_y = (line.y0 + line.y1) / 2
                                diff = abs(line_center_y - cap_center_y)
                                if diff < min_diff:
                                    min_diff = diff
                                    new_closest_line = line
                                    new_closest_group = group
                        if new_closest_group:
                            new_refined_rect = fitz.Rect(
                                min(r.x0 for r in new_closest_group),
                                min(r.y0 for r in new_closest_group),
                                max(r.x1 for r in new_closest_group),
                                max(r.y1 for r in new_closest_group)
                            )
                            pending_rect_draws.append((page.number, new_refined_rect, (1, 0, 1), 2))
                            drawn_rectangles.append((page.number, new_refined_rect))
                            prev_entry = next((entry for entry in drawn_table_regions if entry[0] == page.number and rect_overlap_ratio(entry[1], refined_rect) > 0.8), None)
                            upper_label = prev_entry[2] if prev_entry is not None else cap_label
                            pending_text_inserts.append((page.number, upper_label, (new_refined_rect.x0, new_refined_rect.y0 - 10), (1, 0, 1), 12))
                        else:
                            pending_rect_draws.append((page.number, refined_rect, (0, 1, 0), 2))
                            pending_text_inserts.append((page.number, cap_label, (refined_rect.x0, refined_rect.y0 - 10), (0, 1, 0), 12))
                            drawn_table_regions.append((page.number, refined_rect, cap_label))
                else:
                    pending_rect_draws.append((page.number, refined_rect, (0, 1, 0), 2))
                    pending_text_inserts.append((page.number, cap_label, (refined_rect.x0, refined_rect.y0 - 10), (0, 1, 0), 12))
                    drawn_table_regions.append((page.number, refined_rect, cap_label))
    
        # End of table caption horizontal line handling
    
    # ── 4. 본문 텍스트 블록 중 테이블 영역과 겹치는 부분 제거 ──
    filtered_global_main_blocks = []
    for entry in global_main_blocks:
        page_num, rect, text = entry
        skip = False
        for (pn, table_rect, _) in drawn_table_regions:
            if pn == page_num and rect.intersects(table_rect):
                skip = True
                break
        if not skip:
            filtered_global_main_blocks.append(entry)
    
    # ── 5. 열(컬럼) 검출 (시각화용): 추출된 텍스트 블록들을 그룹화하여 컬럼 영역 표시 ──
    remaining_blocks = filtered_global_main_blocks.copy()
    columns = []
    while remaining_blocks:
        groups = []
        for entry in remaining_blocks:
            pn, rect, _ = entry
            placed = False
            for group in groups:
                _, rep_rect, _ = group[0]
                if abs(rect.x0 - rep_rect.x0) <= GROUP_TOLERANCE and abs(rect.x1 - rep_rect.x1) <= GROUP_TOLERANCE:
                    group.append(entry)
                    placed = True
                    break
            if not placed:
                groups.append([entry])
        dominant = max(groups, key=lambda g: sum(entry[1].get_area() for entry in g))
        columns.append(dominant)
        dom_x0 = min(rect.x0 for (_, rect, _) in dominant)
        dom_x1 = max(rect.x1 for (_, rect, _) in dominant)
        remaining_blocks = [entry for entry in remaining_blocks if not (entry[1].x1 > dom_x0 and entry[1].x0 < dom_x1)]
    
    group_info = []
    for group in columns:
        group_x_min = min(rect.x0 for (_, rect, _) in group)
        group_x_max = max(rect.x1 for (_, rect, _) in group)
        group_width = group_x_max - group_x_min
        group_info.append((group, group_width))
    
    group_info.sort(key=lambda x: x[1], reverse=True)
    
    filtered_groups = []
    if group_info:
        filtered_groups.append(group_info[0])
        for i in range(1, len(group_info)):
            prev_width = filtered_groups[-1][1]
            current_width = group_info[i][1]
            if current_width >= prev_width * 0.9:
                filtered_groups.append(group_info[i])
    
    columns = [grp for (grp, width) in filtered_groups]
    
    for group in columns:
        group_filtered = [entry for entry in group if entry[0] != 0]
        if not group_filtered:
            continue
        group_color = (random.random(), random.random(), random.random())
        group_y_min = min(rect.y0 for (_, rect, _) in group_filtered)
        group_y_max = max(rect.y1 for (_, rect, _) in group_filtered)
        group_x_min = min(rect.x0 for (_, rect, _) in group_filtered)
        group_x_max = max(rect.x1 for (_, rect, _) in group_filtered)
        col_rect = fitz.Rect(group_x_min, group_y_min, group_x_max, group_y_max)
        entire_col_rect |= col_rect
        for entry in group:
            page_num, rect, _ = entry
            pending_main_text_rect_draws.append((page_num, rect, group_color, 1))
            drawn_rectangles.append((page_num, rect))
    
    # ── 6. 이미지 및 드로잉 요소(비텍스트 요소) 클러스터링 및 추출 ──
    merged_clusters_by_page = {}
    for page in doc:
        elements_to_cluster = []
        # 페이지 내 이미지 영역 추출
        for img in page.get_images(full=True):
            xref = img[0]
            img_rects = page.get_image_rects(xref)
            for rect in img_rects:
                if already_drawn(page.number, rect, drawn_rectangles):
                    continue
                skip = False
                for (pn, table_rect, _) in drawn_table_regions:
                    if pn == page.number and rect_overlap_ratio(rect, table_rect) > 0.8:
                        skip = True
                        break
                if skip:
                    continue
                elements_to_cluster.append(rect)
        # 페이지 내 드로잉 요소(라인, 사각형 등) 추출
        for obj in page.get_drawings():
            rect = obj.get("rect")
            skip = False
            if not entire_col_rect.intersects(rect):
                skip = True
            if not skip:
                for (pn, table_rect, _) in drawn_table_regions:
                    if pn == page.number:
                        if rect_overlap_ratio(rect, table_rect) > 0.8:
                            skip = True
                            break
                        if (rect.y1 - rect.y0) < 3:
                            cp = fitz.Point((rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2)
                            if table_rect.contains(cp):
                                skip = True
                                break
                        if (table_rect.contains(rect) or table_rect.intersects(rect)):
                            skip = True
                            break
            if is_in_blocks(page.number, rect, drawn_rectangles):
                skip = True
            if is_intersects_blocks(page.number, rect, drawn_rectangles):
                skip = True
            if skip:
                continue
            elements_to_cluster.append(rect)
    
        # 클러스터링: 가까운 요소들을 그룹화하여 병합 영역 결정
        clusters_rect = cluster_elements(elements_to_cluster, threshold=20)
        merged_cluster_rects = []
        for cluster in clusters_rect:
            merged_rect = fitz.Rect()
            for r in cluster:
                merged_rect |= r
            if merged_rect.width < 5 or merged_rect.height < 5:
                continue
            merged_cluster_rects.append(merged_rect)
        merged_cluster_rects = merge_overlapping_rects(merged_cluster_rects)
        merged_clusters_by_page[page.number] = merged_cluster_rects
    
    # ── 7. 클러스터 영역 내 캡션 재탐지 ──
    # 본문 텍스트 블록과 클러스터 영역이 교차하는 경우, subtract_rect를 통해 후보 영역 분리 후
    # 캡션 패턴에 따라 재탐지 수행
    for entry in filtered_global_main_blocks:
        page_num, text_rect, text = entry
        if page_num not in merged_clusters_by_page:
            continue
        page = doc[page_num]
        for cluster_rect in merged_clusters_by_page[page_num]:
            if text_rect.intersects(cluster_rect):
                candidates = subtract_rect(text_rect, cluster_rect)
                for candidate in candidates:
                    candidate_text = page.get_text("text", clip=candidate).strip()
                    if not candidate_text:
                        continue
                    candidate_text_no_space = re.sub(r'\s+', '', candidate_text)
                    match_fig = fig_pattern.match(candidate_text_no_space)
                    match_table = table_pattern.match(candidate_text_no_space)
                    if match_fig:
                        special_char = match_fig.group("special")
                        if special_char.isalnum() or special_char.isspace():
                            continue
                        fig_label = f"Figure {match_fig.group(2)}"
                        pending_rect_draws.append((page_num, candidate, (1, 0, 0), 2))
                        pending_text_inserts.append((page_num, fig_label, (candidate.x0, candidate.y0), (0, 0, 0), 12))
                        drawn_rectangles.append((page_num, candidate))
                        figure_caption_regions.append((candidate, fig_label, candidate_text, page_num))
                    elif match_table:
                        special_char = match_table.group("special")
                        if special_char.isalnum() or special_char.isspace():
                            continue
                        table_label = f"Table {match_table.group(2)}"
                        pending_rect_draws.append((page_num, candidate, (1, 0, 0), 2))
                        pending_text_inserts.append((page_num, table_label, (candidate.x0, candidate.y0), (0, 0, 0), 12))
                        table_caption_regions.append((candidate, table_label, candidate_text, page_num))
                        drawn_rectangles.append((page_num, candidate))
    
    # ── 8. 텍스트 보강 클러스터링 기능 복원 ──
    # 추가적으로 본문 텍스트 블록을 클러스터링하여 누락된 영역을 보완
    for page in doc:
        elements_to_cluster = []
        for obj in page.get_text("blocks"):
            rect = fitz.Rect(obj[:4])
            skip = False
            if page.number == 0:
                skip = True
            if not entire_col_rect.intersects(rect):
                skip = True
            if not skip:
                for (pn, table_rect, _) in drawn_table_regions:
                    if pn == page.number:
                        if rect_overlap_ratio(rect, table_rect) > 0.8:
                            skip = True
                            break
                        if (rect.y1 - rect.y0) < 3:
                            cp = fitz.Point((rect.x0+rect.x1)/2, (rect.y0+rect.y1)/2)
                            if table_rect.contains(cp):
                                skip = True
                                break
                        if (table_rect.contains(rect) or table_rect.intersects(rect)):
                            skip = True
                            break
                if is_in_blocks(page.number, rect, drawn_rectangles):
                    skip = True
                if is_intersects_blocks(page.number, rect, drawn_rectangles):
                    skip = True
            if skip:
                continue
            elements_to_cluster.append(rect)
        new_clusters = cluster_elements(elements_to_cluster, threshold=20)
        new_cluster_rects = []
        for cluster in new_clusters:
            merged_rect = fitz.Rect()
            for r in cluster:
                merged_rect |= r
            if merged_rect.width < 5 or merged_rect.height < 5:
                continue
            new_cluster_rects.append(merged_rect)
        new_cluster_rects = merge_overlapping_rects(new_cluster_rects)
        if page.number in merged_clusters_by_page:
            merged_clusters_by_page[page.number].extend(new_cluster_rects)
            merged_clusters_by_page[page.number] = merge_overlapping_rects(merged_clusters_by_page[page.number])
        else:
            merged_clusters_by_page[page.number] = new_cluster_rects
    
    # ── 9. 캡션과 클러스터 영역 매칭 (1:1) ──
    # DFS(깊이 우선 탐색) 기반 매칭 알고리즘을 이용하여 캡션과 후보 클러스터를 1:1 매칭
    for page in doc:
        clusters = merged_clusters_by_page.get(page.number, [])
        # 피규어 캡션만 대상으로 매칭 (테이블 캡션은 별도 처리)
        captions_on_page = [(cap_rect, cap_label) for cap_rect, cap_label, _, cap_page in figure_caption_regions if cap_page == page.number]
        if not captions_on_page or not clusters:
            continue
        candidate_clusters = []
        for i, (cap_rect, cap_label) in enumerate(captions_on_page):
            candidates = []
            for j, cluster_rect in enumerate(clusters):
                # 두 영역 사이의 가장 가까운 경계상의 점을 계산
                p_cluster, p_cap = closest_points_between_rectangles(cluster_rect, cap_rect)
                distance = math.hypot(p_cluster[0] - p_cap[0], p_cluster[1] - p_cap[1])
                candidates.append((j, distance, cluster_rect, p_cluster, p_cap))
            candidates.sort(key=lambda x: x[1])
            candidate_clusters.append(candidates)
        match = {}
        def dfs(caption_idx, visited):
            """
            DFS 기반 매칭 알고리즘:
              - 각 캡션 인덱스(candidate_clusters의 인덱스)에 대해
                후보 클러스터 목록을 순회하면서 매칭 가능한 클러스터를 찾음.
              - 이미 매칭된 클러스터에 대해 재귀적으로 다른 캡션과 매칭 가능하면 교체.
            """
            for cand in candidate_clusters[caption_idx]:
                cluster_idx = cand[0]
                if cluster_idx in visited:
                    continue
                visited.add(cluster_idx)
                if cluster_idx not in match or dfs(match[cluster_idx], visited):
                    match[cluster_idx] = caption_idx
                    return True
            return False
        for cap_idx in range(len(captions_on_page)):
            dfs(cap_idx, set())
        for cluster_idx, cap_idx in match.items():
            chosen_candidate = next((cand for cand in candidate_clusters[cap_idx] if cand[0] == cluster_idx), None)
            if chosen_candidate is not None:
                cap_label = captions_on_page[cap_idx][1]
                if page.number not in page_caption_matching:
                    page_caption_matching[page.number] = {}
                page_caption_matching[page.number][cap_label] = (chosen_candidate[2], chosen_candidate[3], chosen_candidate[4], chosen_candidate[1])
    
    # ── 10. 후처리: 매칭되지 않은 클러스터 영역 병합 ──
    # 매칭되지 않은 클러스터 영역 중, 캡션과 충돌하지 않는 영역은 기존 매칭 영역과 병합
    for page in doc:
        if page.number not in merged_clusters_by_page:
            continue
        if page.number not in page_caption_matching:
            continue
        matched_dict = page_caption_matching[page.number]
        matched_list = [(label, tup[0]) for label, tup in matched_dict.items()]
        unmatched = [cl for cl in merged_clusters_by_page[page.number] if not is_in_matched(cl, matched_list)]
        merge_candidates = []
        for cl in unmatched:
            best_distance = float('inf')
            best_label = None
            best_matched = None
            for label, m in matched_list:
                p1, p2 = closest_points_between_rectangles(cl, m)
                distance = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if distance < best_distance:
                    best_distance = distance
                    best_label = label
                    best_matched = m
            if best_label is not None:
                merge_candidates.append((best_distance, best_label, cl))
        merge_candidates.sort(key=lambda x: x[0])
        for dist, label, cl in merge_candidates:
            current_matched = matched_dict[label][0]
            candidate_union = current_matched | cl
            conflict = False
            for pn, dr in drawn_rectangles:
                if pn != page.number:
                    continue
                if candidate_union.intersects(dr) or candidate_union.contains(dr):
                    conflict = True
                    break
            if conflict:
                continue
            for pn, tr, _ in drawn_table_regions:
                if pn != page.number:
                    continue
                if candidate_union.intersects(tr) or candidate_union.contains(tr):
                    conflict = True
                    break
            if conflict:
                continue
            for cap_rect, cap_label, text, cap_page in (figure_caption_regions + table_caption_regions):
                if cap_page != page.number:
                    continue
                if (candidate_union.intersects(cap_rect) or 
                    candidate_union.contains(cap_rect) or 
                    cap_rect.contains(candidate_union)):
                    conflict = True
                    break
            if conflict:
                continue
            matched_dict[label] = (candidate_union, (0,0), (0,0), 0)
            for i, (lbl, m) in enumerate(matched_list):
                if lbl == label:
                    matched_list[i] = (lbl, candidate_union)
                    break
    
    # ── 11. 캡션 영역(테이블 영역)과 클러스터 영역을 별도 PDF로 저장 ──
    # 원본 파일명(확장자 제외)을 사용하여 output 폴더에 저장
    document_name = os.path.splitext(os.path.basename(input_path))[0]
    table_caption_regions = save_regions_as_pdf(doc, table_caption_regions, document_name, drawn_table_regions)
    page_caption_matching = save_cluster_regions_as_pdf(doc, page_caption_matching, document_name)

    # ── 12. 매칭 결과 시각화: PDF에 클러스터 영역 및 캡션 라벨 표시 ──
    for page in doc:
        matching = page_caption_matching.get(page.number, {})
        for cap_label, match_data in matching.items():
            # match_data: (클러스터 사각형, p_cluster, p_cap, 거리)
            cluster_rect = match_data[0]
            page.draw_rect(cluster_rect, (1,0,1), width=5)
            page.insert_text((cluster_rect.x0, cluster_rect.y0 - 10), cap_label, color=(1,0,1), fontsize=12)
    
    # ── 13. pending 리스트에 저장된 그리기 작업 최종 수행 ──
    for (page_num, rect, color, width) in pending_rect_draws:
        doc[page_num].draw_rect(rect, color=color, width=width)
    for (page_num, rect, color, width) in pending_main_text_rect_draws:
        doc[page_num].draw_rect(rect, color=color, width=width)
    for (page_num, text, pos, color, fontsize) in pending_text_inserts:
        doc[page_num].insert_text(pos, text, color=color, fontsize=fontsize)
    for (page_num, p1, p2, color, width) in pending_line_draws:
        doc[page_num].draw_line(p1, p2, color=color, width=width)
    
    doc.save(output_path)
    doc.close()
    
    # ── 최종 SQL 저장: PDF 처리 후 모든 결과 정보를 MySQL 데이터베이스에 저장 ──
    save_pdf_id = save_to_sql(os.path.basename(input_path), table_caption_regions, figure_caption_regions, drawn_table_regions, page_caption_matching)
    print(f"Processed and saved: {os.path.basename(input_path)}")
    return save_pdf_id
def main():
    """
    main 함수:
      - 지정한 input 디렉터리 내의 모든 PDF 파일을 순차적으로 처리하여 output 디렉터리에 저장.
      - 각 파일 처리 후 로그 출력.
    """
    input_dir = "data"
    output_dir = "clustered"
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            save_pdf_id = process_pdf(input_path, output_path)
            print(f"Processed: {filename}")
            print(f"save_pdf_id: {save_pdf_id}")
    print("모든 파일 처리 완료!")

if __name__ == '__main__':
    main()
