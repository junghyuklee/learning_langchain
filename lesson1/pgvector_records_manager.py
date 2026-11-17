from langchain_classic.indexes import SQLRecordManager
from langchain_core.indexing import index
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from db_config import PG_CONNECTION

connection = PG_CONNECTION
collection_name = "my_docs"
embeddings_model = OllamaEmbeddings(model="bge-m3")
namespace = "my_docs_namespace"

vectorstore = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

record_manager = SQLRecordManager(
    namespace,
    db_url=PG_CONNECTION,
)

# 스키마가 없으면 생성
record_manager.create_schema()

# 문서 생성
docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "source": "cats.txt"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "source": "ducks.txt"},
    ),
]

# 문서 인덱싱 1회차
index_1 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",  # 문서 중복 방지
    source_id_key="source",  # 출처를 source_id로 사용
)

print("인덱싱 1회차:", index_1)

# 문서 인덱싱 2회차, 중복 문서 생성 안 됨
index_2 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("인덱싱 2회차:", index_2)

# 문서를 수정하면 새 버전을 저장하고, 출처가 같은 기존 문서는 삭제

docs[0].page_content = "I just modified this document!"

index_3 = index(
    docs,
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)

print("인덱싱 3회차:", index_3)
