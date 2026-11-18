from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_postgres.vectorstores import PGVector
from langchain_postgres import PGVectorTranslator
from db_config import PG_CONNECTION
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.structured_query import Comparator


# 기본 PGVectorTranslator는 gte/lte를 허용 목록에서 빼놨고, PGVector가 지원하지 않는 contain은 넣어놨음
# → PGVector가 실제로 지원하는 비교 연산자로 교정
class FixedPGVectorTranslator(PGVectorTranslator):
    allowed_comparators = [
        Comparator.EQ,
        Comparator.NE,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
        Comparator.IN,
        Comparator.NIN,
        Comparator.LIKE,
    ]


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

# 문서 임베딩 생성성
embeddings_model = OllamaEmbeddings(model="bge-m3")

# pre_delete_collection=True: 실행할 때마다 같은 문서가 중복 저장되지 않도록 컬렉션을 비우고 다시 넣음
vectorstore = PGVector.from_documents(
    docs,
    embeddings_model,
    connection=PG_CONNECTION,
    collection_name="movies",
    pre_delete_collection=True,
)

fields = [
    AttributeInfo(name="genre", description="영화 장르", type="string"),
    AttributeInfo(name="year", description="영화 개봉 연도", type="integer"),
    AttributeInfo(name="director", description="영화 감독", type="string"),
    AttributeInfo(name="rating", description="영화 평점 1-10점", type="float"),
]

description = "영화에 대한 간략한 정보"
llm = ChatOllama(model="qwen2.5:14b", temperature=0)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    description,
    fields,
    structured_query_translator=FixedPGVectorTranslator(),
)

# 필터 적용
print(retriever.invoke("평점이 8.5점 이상인 영화가 보고 싶어요."))

print("\n")

# 다양한 필터 적용
print(retriever.invoke("평점이 높은(8.5 이상) SF영화는 무엇인가요?"))
