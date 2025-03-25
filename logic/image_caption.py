# image_caption.py
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed

# BLIP 모델과 프로세서를 전역으로 초기화 (모듈이 처음 import 될 때 한 번만 로딩됨)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    """
    주어진 이미지 파일 경로에 대해 캡션을 생성합니다.
    :param image_path: 이미지 파일 경로
    :return: (파일명, 캡션) 튜플
    """
    try:
        image = Image.open(image_path)
        inputs = processor(image, return_tensors="pt")
        # 생성 옵션: max_length, num_beams=1로 빠른 추론 유도
        caption_ids = model.generate(**inputs, max_length=50, num_beams=1)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        return os.path.basename(image_path), caption
    except Exception as e:
        return os.path.basename(image_path), f"Error: {e}"

def get_image_captions(png_path, max_workers=4):
    """
    지정된 경로에 대해 캡션을 생성합니다.
    - 만약 png_path가 디렉터리라면, 해당 폴더 내의 모든 PNG 파일에 대해 캡션을 생성하고,
      결과를 dict(파일명 -> 캡션)로 반환합니다.
    - 만약 png_path가 단일 PNG 파일이라면, 해당 파일에 대해 캡션을 생성하고 캡션 문자열을 반환합니다.
    
    :param png_path: PNG 파일 또는 PNG 파일들이 위치한 폴더 경로
    :param max_workers: 병렬 처리에 사용할 스레드 수 (기본 4)
    :return: dict 또는 단일 캡션 문자열
    """
    # png_path가 디렉터리인 경우
    if os.path.isdir(png_path):
        captions = {}
        png_files = [os.path.join(png_path, f) for f in os.listdir(png_path) if f.lower().endswith(".png")]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(generate_caption, file): file for file in png_files}
            for future in as_completed(futures):
                filename, caption = future.result()
                captions[filename] = caption
                print(f"File: {filename}, Caption: {caption}")
        return captions

    # png_path가 단일 PNG 파일인 경우
    elif os.path.isfile(png_path) and png_path.lower().endswith(".png"):
        filename, caption = generate_caption(png_path)
        print(f"File: {filename}, Caption: {caption}")
        return caption

    else:
        raise ValueError(f"{png_path}는(은) 유효한 PNG 파일 또는 디렉터리가 아닙니다.")

if __name__ == '__main__':
    # 모듈 단독 실행 시 테스트: "png" 폴더 또는 단일 PNG 파일에 대해 캡션을 생성하고 출력
    # 테스트 경로를 필요에 따라 수정하세요.
    test_path = "png"  # 예: 폴더 "png" 내의 모든 PNG 파일 처리
    # test_path = "png/example.png"  # 예: 단일 PNG 파일 처리
    result = get_image_captions(test_path)
    
    if isinstance(result, dict):
        print("\n전체 캡션 결과:")
        for file, cap in result.items():
            print(f"{file}: {cap}")
    else:
        print("\n캡션 결과:", result)
