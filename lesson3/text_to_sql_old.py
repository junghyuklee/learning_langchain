from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from db_config import PG_CONNECTION

# 사용할 db 경로 수정
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.get_usable_table_names())

llm = ChatOllama(model="qwen2.5:14b", temperature=0)

# 질문을 SQL 쿼리로 변환
write_query = create_sql_query_chain(llm, db)

# SQL 쿼리 실행
execute_query = QuerySQLDatabaseTool(db=db)

# LLM이 "SQLQuery: SELECT ..."처럼 접두어를 붙여 출력하는 경우가 있어 제거 후 실행
combined_chain = (
    write_query
    | (lambda q: q.strip().removeprefix("SQLQuery:").strip())
    | execute_query
)

# 체인 실행
result = combined_chain.invoke({"question": "직원(employee)은 모두 몇 명인가요?"})
print(result)
