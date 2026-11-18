from langchain_ollama import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from db_config import PG_CONNECTION
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import chain

llm = ChatOllama(model="qwen2.5:14b", temperature=0)

embeddings_model = OllamaEmbeddings(model="bge-m3")
db = PGVector(
    embeddings=embeddings_model,
    collection_name="langchain",  # from_documents 기본 컬렉션 이름
    connection=PG_CONNECTION,
)

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt_hyde = ChatPromptTemplate.from_template("""
질문에 답할 구절을 작성해 주세요.
질문: {question}

구절:""")


generate_doc = prompt_hyde | llm | StrOutputParser()

query = "인공지능의 발전으로 인한 장점이 뭐가 있어?"

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트:{context}

질문: {question}
""")


@chain
def qa(input):
    # 1) HyDE: 질문에 대한 가상의 답변 문서 생성
    hyde_doc = generate_doc.invoke(input)
    print("=== HyDE로 생성된 가상 문서 ===")
    print(hyde_doc)
    print()

    # 2) 질문 대신 가상 문서로 검색
    docs = retriever.invoke(hyde_doc)
    print("=== 검색된 실제 문서 ===")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content[:100]}...")
    print()

    # 3) 검색된 실제 문서로 답변 생성
    context = "\n\n".join(d.page_content for d in docs)
    formatted = prompt.invoke({"context": context, "question": input})
    answer = llm.invoke(formatted)
    return answer


print("HyDE 실행\n")
result = qa.invoke(query)
print("=== 최종 답변 ===")
print(result.content)
