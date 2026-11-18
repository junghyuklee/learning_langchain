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

# 내부 검색 데이터 활용을 위해 창의성을 배제하도록 temperature 를 0으로 설정
llm = ChatOllama(model="qwen2.5:14b", temperature=0)

prompt_rag_fusion = ChatPromptTemplate.from_template(
    """하나의 입력 쿼리를 기반으로 여러개의 검색 쿼리를 생성하는 유용한 어시스턴트입니다. 다음과 관련된 여러 검색 쿼리를 생성합니다.
번호나 설명 없이, 검색 쿼리 4개를 한 줄에 하나씩 출력하세요.

입력 쿼리: {question}"""
)

query = "인공지능의 발전으로 인한 부작용이 뭐가 있어?"


def parse_queries_output(message):
    return message.content.split("\n")


query_gen = prompt_rag_fusion | llm | parse_queries_output


def reciprocal_rank_fusion(results: list[list], k=60):
    """여러 순위 문서 목록에 대한 상호 순위 융합 및 RRF 공식에 사용되는 선택적 매개변수 k입니다."""
    # 사전을 초기화해 각 문서에 대한 융합된 점수를 보관합니다.
    # 공유성를 보장하기 위해 문서가 콘텐츠별로 키를 만듭니다.
    fused_scores = {}
    documents = {}
    for docs in results:
        # 목록에 있는 각 문서를 순위(목록 내 위치)에 따라 반복
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)
        # 융합된 점수를 기준으로 문서를 내림차순으로 정렬하여 최종 재순위 결과를 정리
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True
    )
    return [documents[doc_str] for doc_str in reranked_doc_strs]


retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트만 사용해 질문에 답하세요.
컨텍스트:{context}

질문: {question}
""")


@chain
def multi_query_qa(input):
    docs = retrieval_chain.invoke(input)
    context = "\n\n".join(d.page_content for d in docs)
    answer = llm.invoke(prompt.invoke({"context": context, "question": input}))
    return answer


print("다중 쿼리 검색\n")
result = multi_query_qa.invoke(query)
print(result.content)
