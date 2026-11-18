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

perspectives_prompt = ChatPromptTemplate.from_template(
    """당신은 AI 언어 모델 어시스턴트입니다. 주어진 사용자 질문의 다섯 가지 버전을 생성하여 백터 데이터베이스에서 관련 문서를 검색하세요. 사용자 질문에 대한 다양한 관점을 생성함으로써 사용자가 거리기반 유사도 검색의 한계를 극복할 수 있도록 돕는 것이 목표입니다. 이러한 대체 질문을 개행으로 구분하여 제공하세요. 원래질문: {question}"""
)


# 내부 검색 데이터 활용을 위해 창의성을 배제하도록 temperature 를 0으로 설정
llm = ChatOllama(model="qwen2.5:14b", temperature=0)


def parse_rewriter_output(message):
    return message.content.strip("'").strip("**")


def parse_queries_output(message):
    queries = message.content.split("\n")
    # 생성된 쿼리 확인용 출력 (빈 줄도 그대로 보이도록 repr 사용)
    print(f"[생성된 쿼리 {len(queries)}개]")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q!r}")
    print()
    return queries


query_gen = perspectives_prompt | llm | parse_queries_output


def get_unique_union(document_lists):
    # 목록 여러 개를 포함한 리스트를 평탄화하고 중복 제거
    deduped_docs = {
        doc.page_content: doc for sublist in document_lists for doc in sublist
    }
    # 고유한 문서만 반환
    return list(deduped_docs.values())


retrieval_chain = query_gen | retriever.batch | get_unique_union


prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트:{context}

질문: {question}
""")

query = "인공지능의 발전으로 인한 부작용이 뭐가 있어?"


@chain
def multi_query_qa(input):
    docs = retrieval_chain.invoke(input)
    context = "\n\n".join(d.page_content for d in docs)
    answer = llm.invoke(prompt.invoke({"context": context, "question": input}))
    return answer


print("다중 쿼리 검색\n")
result = multi_query_qa.invoke(query)
print(result.content)
