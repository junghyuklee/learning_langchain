"""PostgreSQL(pgvector) 연결 정보.

프로젝트 루트의 .env 값을 읽어 LangChain(PGVector)용 연결 문자열을 만든다.

사용 예:
    from db_config import PG_CONNECTION
    from langchain_postgres import PGVector

    store = PGVector(embeddings=..., collection_name="docs", connection=PG_CONNECTION)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

PG_USER = os.getenv("PG_USER", "langchain")
PG_PASSWORD = os.getenv("PG_PASSWORD", "langchain")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "langchain")

# langchain-postgres(PGVector)용: SQLAlchemy + psycopg3 드라이버
PG_CONNECTION = (
    f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)


if __name__ == "__main__":
    # 비밀번호는 가려서 출력
    print(PG_CONNECTION.replace(f":{PG_PASSWORD}@", ":****@"))
