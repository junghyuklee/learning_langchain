"""
Self-Query를 요즘 방식으로 구현한 예제.

SelfQueryRetriever(langchain_classic)는 LLM이 'and(gte("rating", 8.5), ...)' 같은
자체 문법의 문자열을 쓰게 하고, lark로 파싱한 뒤 벡터DB별 Translator로 변환한다.
→ 허용 연산자 누락, 문자열/리스트 혼동 같은 오류가 런타임에야 드러난다.

이 파일은 대신:
  1. LLM이 Pydantic 스키마(MovieSearch)를 채우게 하고 (with_structured_output)
  2. 어떤 연산자를 쓸지는 내 코드(build_filter)가 정한다.
"""

from typing import Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector
from pydantic import BaseModel, Field

from db_config import PG_CONNECTION

docs = [
    Document(
        page_content="과학자들이 공룡을 되살리고 대혼란이 일어난다.",
        metadata={"year": 1993, "rating": 8.7, "genre": "SF"},
    ),
    Document(
        page_content="레오나르도 디카프리오가 꿈속의 꿈속의 꿈속의 꿈속에 빠진다.",
        metadata={"year": 2010, "director": "크리스토퍼 놀란", "rating": 8.2},
    ),
    Document(
        page_content="심리학자인 형사가 꿈속의 꿈속의 꿈속의 꿈속의 꿈속에 빠진다. 인셉션이 이 발상을 차용했다.",
        metadata={"year": 2006, "director": "곤 사토시", "rating": 8.6},
    ),
    Document(
        page_content="평범한 체형의 매우 건강하고 순수한 매력을 지닌 여성들을 남성들이 동경한다.",
        metadata={"year": 2019, "director": "그레타 거윅", "rating": 8.3},
    ),
    Document(
        page_content="장난감들이 살아 움직이며 신나는 시간을 보낸다",
        metadata={"year": 1995, "genre": "애니메이션"},
    ),
    Document(
        page_content="세 남자가 구역으로 들어가고, 세 남자가 구역 밖으로 나온다.",
        metadata={
            "year": 1979,
            "director": "안드레이 타르코프스키",
            "genre": "스릴러",
            "rating": 9.9,
        },
    ),
]

embeddings_model = OllamaEmbeddings(model="bge-m3")

# pre_delete_collection=True: 실행할 때마다 같은 문서가 중복 저장되지 않도록 컬렉션을 비우고 다시 넣음
vectorstore = PGVector.from_documents(
    docs,
    embeddings_model,
    connection=PG_CONNECTION,
    collection_name="movies",
    pre_delete_collection=True,
)


# 1. LLM이 채울 검색 스키마
#    - 필드 설명(description)이 곧 LLM에게 주는 지시문
#    - genre를 Literal로 제한하면 LLM이 DB에 없는 장르명("공상과학" 등)을 만들어내지 못함
class MovieSearch(BaseModel):
    """영화 검색 조건"""

    query: str = Field(
        description="영화 줄거리/내용과 의미적으로 비교할 검색어. 필터 조건(평점, 연도, 장르, 감독)은 빼고 내용만 남긴다. 내용 조건이 없으면 빈 문자열."
    )
    genre: Literal["SF", "애니메이션", "스릴러"] | None = Field(
        default=None, description="영화 장르"
    )
    director: str | None = Field(default=None, description="감독 이름")
    min_rating: float | None = Field(default=None, description="최소 평점 (1-10, 이상)")
    max_rating: float | None = Field(default=None, description="최대 평점 (1-10, 이하)")
    min_year: int | None = Field(default=None, description="개봉 연도 하한 (이상)")
    max_year: int | None = Field(default=None, description="개봉 연도 상한 (이하)")


llm = ChatOllama(model="qwen2.5:14b", temperature=0)

# 스키마 설명만으로는 로컬 모델이 필드를 놓치거나 없는 조건을 지어내는 경우가 많음
# → 규칙 + few-shot 예시를 함께 주는 것이 query analysis의 일반적인 방법
system = """당신은 영화 검색 질문을 분석해 검색 조건(JSON)으로 변환하는 전문가입니다.

규칙:
- 질문에 명시된 조건만 채운다. 언급되지 않은 필드는 반드시 null.
- 장르(SF, 애니메이션, 스릴러)가 언급되면 genre에 넣는다.
- 감독 이름이 언급되면 director에 넣는다.
- "N 이상"은 min_*, "N 이하"는 max_*, "N년 이전"은 max_year=N-1, "N년 이후"는 min_year=N.
- query에는 줄거리/내용 키워드만 남긴다. 장르·감독·평점·연도는 query에서 뺀다.

예시:
질문: 90년대 스릴러 중 평점 9점 이상
→ {{"query": "", "genre": "스릴러", "director": null, "min_rating": 9.0, "max_rating": null, "min_year": 1990, "max_year": 1999}}

질문: 봉준호 감독의 가족 이야기
→ {{"query": "가족 이야기", "genre": null, "director": "봉준호", "min_rating": null, "max_rating": null, "min_year": null, "max_year": null}}"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

query_analyzer = prompt | llm.with_structured_output(MovieSearch)


# 2. 스키마 → PGVector 필터 변환 (연산자는 코드가 결정)
def build_filter(search: MovieSearch) -> dict | None:
    conditions = []
    if search.genre:
        conditions.append({"genre": {"$eq": search.genre}})
    if search.director:
        # 감독 이름은 부분 일치 (예: "놀란" → "크리스토퍼 놀란")
        conditions.append({"director": {"$ilike": f"%{search.director}%"}})
    if search.min_rating is not None:
        conditions.append({"rating": {"$gte": search.min_rating}})
    if search.max_rating is not None:
        conditions.append({"rating": {"$lte": search.max_rating}})
    if search.min_year is not None:
        conditions.append({"year": {"$gte": search.min_year}})
    if search.max_year is not None:
        conditions.append({"year": {"$lte": search.max_year}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# 3. 질문 분석 → 필터 검색을 하나의 Runnable로 묶음
@chain
def movie_retriever(question: str) -> list[Document]:
    search = query_analyzer.invoke({"question": question})
    flt = build_filter(search)
    print(f"  분석 결과: {search}")
    print(f"  필터: {flt}")

    # 내용 조건이 없으면 원래 질문으로 유사도 검색 (필터가 실질적인 조건 역할)
    return vectorstore.similarity_search(search.query or question, k=4, filter=flt)


if __name__ == "__main__":
    questions = [
        "평점이 8.5점 이상인 영화가 보고 싶어요.",
        "평점이 높은(8.5 이상) SF영화는 무엇인가요?",
        "놀란 감독의 꿈에 관한 영화",
        "2000년 이전에 나온 애니메이션",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        for doc in movie_retriever.invoke(q):
            print(f"  - {doc.page_content} {doc.metadata}")
