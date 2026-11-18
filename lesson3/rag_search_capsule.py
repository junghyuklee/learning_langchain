from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from db_config import PG_CONNECTION
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

embeddings_model = OllamaEmbeddings(model="bge-m3")
db = PGVector(
    embeddings=embeddings_model,
    collection_name="langchain",  # from_documents 기본 컬렉션 이름
    connection=PG_CONNECTION,
)

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    "아래 컨텍스트만 참고해서 질문에 답하세요. 모르면 모른다고 답하세요.\n\n"
    "컨텍스트:\n{context}\n\n질문: {question}"
)
query = "인공지능의 환각현상 대처방안은?"
# 내부 검색 데이터 활용을 위해 창의성을 배제하도록 temperature 를 0으로 설정
llm = ChatOllama(model="qwen2.5:14b", temperature=0)


@chain
def qa(input):
    # 관련 문서 검색
    docs = retriever.invoke(input)
    # 원문만 이어붙이는게 깔끔
    context = "\n\n".join(d.page_content for d in docs)
    # 답변 생성
    answer = (prompt | llm).invoke({"context": context, "question": input})
    return {"answer": answer.content, "context": context, "docs": docs}


result = qa.invoke(query)
print("=== answer ===")
print(result["answer"])

print("\n=== context ===")
print(result["context"][:500], "...")

print("\n=== docs ===")
for i, d in enumerate(result["docs"], 1):
    print(f"[{i}] {d.metadata.get('source')} | {d.page_content[:80]}...")
