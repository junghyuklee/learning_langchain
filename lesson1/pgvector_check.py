"""pgvector에 저장된 임베딩 확인 스크립트."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_config import PG_DB, PG_HOST, PG_PASSWORD, PG_PORT, PG_USER

import psycopg

with psycopg.connect(
    host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
) as conn:
    print("[컬렉션별 저장 개수]")
    rows = conn.execute(
        """
        SELECT c.name, count(e.id)
        FROM langchain_pg_collection c
        LEFT JOIN langchain_pg_embedding e ON e.collection_id = c.uuid
        GROUP BY c.name
        """
    ).fetchall()
    for name, cnt in rows:
        print(f"  {name}: {cnt}개")

    print("\n[샘플 3개]")
    rows = conn.execute(
        """
        SELECT left(replace(document, E'\\n', ' '), 60),
               vector_dims(embedding),
               cmetadata
        FROM langchain_pg_embedding
        LIMIT 3
        """
    ).fetchall()
    for doc, dims, meta in rows:
        print(f"  - ({dims}차원) {doc}...")
        print(f"    metadata: {meta}")
