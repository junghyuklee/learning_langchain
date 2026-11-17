from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from db_config import PG_CONNECTION
import uuid
from langchain_core.documents import Document

embeddings_model = OllamaEmbeddings(model="bge-m3")
db = PGVector(
    embeddings=embeddings_model,
    collection_name="langchain",  # from_documents 기본 컬렉션 이름
    connection=PG_CONNECTION,
)

ids = [str(uuid.uuid4()), str(uuid.uuid4())]
db.add_documents(
    (
        Document(
            page_content="there are cats in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
        Document(
            page_content="ducks are also found in the pond",
            metadata={"location": "pond", "topic": "animals"},
        ),
    ),
    ids=ids,
)
