from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from db_config import PG_CONNECTION
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

embeddings_model = OllamaEmbeddings(model="bge-m3")
db = PGVector(
    embeddings=embeddings_model,
    collection_name="langchain",  # from_documents 기본 컬렉션 이름
    connection=PG_CONNECTION,
)

retriever = db.as_retriever(search_kwargs={"k": 3})
query = "인공지능의 환각현상 대처방안은?"

prompt = ChatPromptTemplate.from_template(
    "아래 컨텍스트만 참고해서 질문에 답하세요. 모르면 모른다고 답하세요.\n\n"
    "컨텍스트:\n{context}\n\n질문: {question}"
)
# 내부 검색 데이터 활용을 위해 창의성을 배제하도록 temperature 를 0으로 설정
llm = ChatOllama(model="qwen2.5:14b", temperature=0)

docs = retriever.invoke(query)
context = "\n\n".join(d.page_content for d in docs)
answer = (prompt | llm).invoke({"context": context, "question": query})
print(answer.content)
