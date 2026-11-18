from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from db_config import PG_CONNECTION

# 사용할 db 경로 수정
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.get_usable_table_names())

llm = ChatOllama(model="qwen2.5:14b", temperature=0)

# 질문을 SQL 쿼리로 변환
tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

agent = create_agent(
    llm,
    tools,
    system_prompt="""SQLite DB에 질문하는 에이전트입니다.
먼저 테이블 목록과 스키마를 확인하고, 쿼리를 검사한 뒤 실행하세요.
에러가 나면 쿼리를 고쳐 다시 시도하세요. INSERT/UPDATE/DELETE는 절대 금지.""",
)

# 체인 실행
result = agent.invoke({"question": "직원(employee)은 모두 몇 명인가요?"})
print(result)
