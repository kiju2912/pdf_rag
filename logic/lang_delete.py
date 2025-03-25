from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from dotenv import load_dotenv
import os
def initialize_model():
    # 환경변수 로드 (필요한 경우)
    load_dotenv()

    # 임베딩 모델 설정
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    # Milvus 서버 연결 정보
    milvus_uri = "tcp://localhost:19530"

    # 기존 컬렉션을 삭제(drop_old=True)하고 새로 생성하여 데이터를 초기화함
    vectorstore = Milvus(
        embedding,
        connection_args={"uri": milvus_uri},
        collection_name="docling_demo",
        drop_old=True,  # 기존 데이터를 삭제하고 컬렉션 초기화
        index_params={"index_type": "FLAT", "metric_type": "L2", "params": {}},
    )

    print("Milvus 컬렉션이 초기화되었습니다.")
