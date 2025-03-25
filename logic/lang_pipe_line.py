# rag_pipeline.py
import os
import json
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
import mysql.connector

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.documents import Document
import config
# 환경변수 설정 및 토큰 로드
load_dotenv()
HF_TOKEN = os.getenv(config.HF_TOKEN)
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
print("환경변수 및 HuggingFace 토큰 로드 완료")

# TOKENIZERS 병렬 처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_pdf_filename_from_sql(pdf_id: int) -> str:
    """
    SQL 데이터베이스에서 주어진 pdf_id에 해당하는 file_name을 조회합니다.
    """
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='00000000',
        database='pdf_parser',
        port=3306
    )
    cursor = conn.cursor()
    query = "SELECT file_name FROM pdf_documents WHERE pdf_id = %s"
    cursor.execute(query, (pdf_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result is None:
        raise ValueError(f"pdf_id {pdf_id}에 해당하는 파일이 존재하지 않습니다.")
    return result[0]

def get_all_sql_metadata(pdf_id: int) -> list:
    """
    SQL 데이터베이스에서 주어진 pdf_id에 해당하는 모든 캡션 메타데이터를 조회합니다.
    captions 테이블과 area 테이블을 LEFT JOIN하여 appearance_description 및 pdf_file_name 값을 함께 가져옵니다.
    """
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='00000000',
        database='pdf_parser',
        port=3306
    )
    cursor = conn.cursor()
    query = """
    SELECT 
      c.caption_id, 
      c.caption_name, 
      c.pdf_id, 
      c.page_number, 
      c.caption_text, 
      c.x0, c.y0, c.x1, c.y1,
      a.appearance_description,
      a.pdf_file_name
    FROM captions c
    LEFT JOIN area a ON c.caption_id = a.caption_id
    WHERE c.pdf_id = %s
    """
    cursor.execute(query, (pdf_id,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    metadata_list = []
    for row in results:
        metadata_list.append({
            "caption_id": row[0],
            "caption_name": row[1],
            "pdf_id": str(row[2]),
            "page_number": str(row[3]),
            "caption_text": row[4],
            "x0": row[5],
            "y0": row[6],
            "x1": row[7],
            "y1": row[8],
            "appearance_description": row[9],
            "pdf_file_name": row[10]
        })
    return metadata_list

def build_rag_pipeline_for_pdf_id(pdf_id: int) -> object:
    """
    SQL에서 pdf_id를 통해 file_name과 관련된 모든 캡션 메타데이터(appearance_description 포함)를 조회한 후,
    해당 PDF 파일(data/(file_name))을 DoclingLoader로 로딩·청킹하고,
    SQL 메타데이터를 기반으로 Document 객체들을 생성하여 추가 문서로 합친 후,
    Milvus 벡터스토어와 RAG 체인을 구성한 후, 구축된 RAG 체인 객체를 반환합니다.
    """
    # 1. SQL에서 file_name 조회
    file_name = get_pdf_filename_from_sql(pdf_id)
    pdf_path = os.path.join("data", file_name)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"{pdf_path} 파일이 존재하지 않습니다.")
    print(f"PDF 파일 로딩 시작: {pdf_path}")
    
    # 2. DoclingLoader를 사용한 PDF 로딩 및 청킹
    EXPORT_TYPE = ExportType.DOC_CHUNKS  # 필요에 따라 ExportType.MARKDOWN 선택 가능
    print("DoclingLoader를 통한 문서 로딩 시작")
    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=EXPORT_TYPE,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = loader.load()
    print(f"DoclingLoader 로딩 완료: {len(docs)} 문서 로드됨")
    
    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docs
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        print("Markdown 형식의 청크 생성을 시작합니다.")
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        splits = [chunk for doc in docs for chunk in splitter.split_text(doc.page_content)]
        print("Markdown 청크 생성 완료")
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")
    
    print(f"문서 청킹 완료: {len(splits)} 청크 생성됨")
    
    # 3. SQL에서 해당 pdf_id와 관련된 모든 캡션 메타데이터 조회 및 Document 객체 생성
    metadata_list = get_all_sql_metadata(pdf_id)
    additional_docs = []
    for meta in metadata_list:
        doc_content = f'''
            "caption_id": "{meta["caption_id"]}",
            "caption_name": "{meta["caption_name"]}",
            "pdf_id": "{meta["pdf_id"]}",
            "page_number": "{meta["page_number"]}",
            "caption_text": "{meta["caption_text"]}",
            "x0": {meta["x0"]},
            "y0": {meta["y0"]},
            "x1": {meta["x1"]},
            "y1": {meta["y1"]},
            "appearance_description": "{meta["appearance_description"]}",
            "pdf_file_name": "{meta["pdf_file_name"]}"
        '''
        additional_docs.append(
            Document(
                page_content=doc_content,
                metadata={
                    "source": "sql",
                    "dl_meta": "",
                    "caption_id": meta["caption_id"],
                    "caption_name": meta["caption_name"],
                    "pdf_id": meta["pdf_id"],
                    "page_number": meta["page_number"],
                    "caption_text": meta["caption_text"],
                    "x0": meta["x0"],
                    "y0": meta["y0"],
                    "x1": meta["x1"],
                    "y1": meta["y1"],
                    "appearance_description": meta["appearance_description"],
                    "pdf_file_name": meta["pdf_file_name"]
                }
            )
        )
    print(f"SQL 메타데이터 기반 추가 Document 객체 생성 완료: {len(additional_docs)}개")
    
    # 4. 벡터스토어 구축 (PDF 청킹 결과와 추가 문서를 합쳐서)
    combined_docs = splits + additional_docs
    TOP_K = 3
    print("임베딩 모델 생성 시작")
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    print("임베딩 모델 생성 완료")
    
    milvus_uri = "tcp://localhost:19530"
    print(f"Milvus 데이터베이스 생성 위치: {milvus_uri}")
    collection_name = f"docling_demo_{pdf_id}"
    vectorstore = Milvus.from_documents(
        documents=combined_docs,
        embedding=embedding,
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
        index_params={
            "index_type": "FLAT",
            "metric_type": "L2",
            "params": {}
        },
        drop_old=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # 5. 언어 모델 초기화 및 RAG 체인 생성
    GEN_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print("HuggingFaceEndpoint를 통한 언어 모델 초기화 시작")
    llm = HuggingFaceEndpoint(
        repo_id=GEN_MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
    )
    print("언어 모델 초기화 완료")
    
    PROMPT = PromptTemplate.from_template(
        "Context information is below.\n---------------------\n{context}\n---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Query: {input}\nAnswer:\n"
    )
    print("질문-응답 체인 생성 시작")
    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    print("질문-응답 체인 생성 완료")
    
    return rag_chain

def execute_question(rag_chain: object, question: str) -> dict:
    """
    구축된 RAG 체인을 사용하여 질문을 실행하고 결과를 반환합니다.
    
    Parameters:
      - rag_chain: build_rag_pipeline_for_pdf_id()로 생성한 RAG 체인 객체
      - question: 사용자 질문 문자열
      
    Returns:
      - 질문 실행 결과 (dict)
    """
    print("질문 실행 시작")
    resp_dict = rag_chain.invoke({"input": question})
    print("질문 실행 완료")
    return resp_dict

# 헬퍼 함수: Document 객체를 직렬화 가능한 dict로 변환
def serialize_document(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

if __name__ == '__main__':
    # 모듈 단독 테스트용 (예시)
    test_pdf_id = 325
    test_question = "explain Figure 4"
    rag_chain = build_rag_pipeline_for_pdf_id(test_pdf_id)
    response = execute_question(rag_chain, test_question)
    # context 내 Document 객체 직렬화
    if "context" in response:
        response["context"] = [serialize_document(doc) for doc in response["context"]]
    print("응답 결과:")
    print(json.dumps(response, indent=2, ensure_ascii=False))
