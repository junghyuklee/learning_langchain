from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from db_config import PG_CONNECTION

# 저장할 때와 같은 임베딩 모델 / 컬렉션 이름을 써야 검색이 맞게 동작
embeddings_model = OllamaEmbeddings(model="bge-m3")

db = PGVector(
    embeddings=embeddings_model,
    collection_name="langchain",  # from_documents 기본 컬렉션 이름
    connection=PG_CONNECTION,
)

results = db.similarity_search_with_score("접근법", k=3)
for doc, score in results:
    print(f"[거리 {score:.4f}] {doc.page_content[:100]}...\n")
