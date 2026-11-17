from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
import sys
import uuid
from pathlib import Path

# 프로젝트 루트(db_config.py 위치)를 import 경로에 추가 → 어느 폴더에서 실행해도 동작
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from db_config import PG_CONNECTION

connection = PG_CONNECTION

raw_documents = TextLoader(str(HERE / "test.txt"), encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documents = text_splitter.split_documents(raw_documents)

embeddings_model = OllamaEmbeddings(model="bge-m3")

db = PGVector.from_documents(
    documents, embeddings_model, connection=connection, pre_delete_collection=True
)
