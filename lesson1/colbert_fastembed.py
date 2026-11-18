"""ColBERT(late interaction) 검색 실습 - Windows에서 동작하는 fastembed 버전.

coibert_test.py(RAGatouille, Windows 미지원)와 같은 흐름:
    위키백과 문서 로드 → 분할 → ColBERT 인덱싱 → 검색 → LangChain retriever로 사용

설치:
    pip install fastembed

ColBERT 핵심 아이디어:
    - 일반 임베딩: 문장 전체를 벡터 1개로 → 코사인 유사도 1번
    - ColBERT: 토큰마다 벡터를 만듦 (문서: N×128, 질문: M×128)
      점수 = 질문의 각 토큰마다 "가장 비슷한 문서 토큰"과의 유사도를 구해 합산 (MaxSim)
      → 단어 단위로 세밀하게 비교하므로 검색 정확도가 높음
"""

import numpy as np
import requests
from fastembed import LateInteractionTextEmbedding
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 영어 문서용. 한국어 문서는 "jinaai/jina-colbert-v2" 또는 "lightonai/mLateOn"
MODEL_NAME = "colbert-ir/colbertv2.0"


def get_wikipedia_page(title: str) -> str | None:
    """위키백과 페이지 전체 텍스트를 가져온다."""
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
        headers={"User-Agent": "learning-langchain/0.1"},
        timeout=30,
    )
    page = next(iter(response.json()["query"]["pages"].values()))
    return page.get("extract")


def maxsim(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    """ColBERT 점수: 질문 토큰마다 가장 비슷한 문서 토큰의 유사도를 합산."""
    sim = query_vec @ doc_vec.T  # (질문 토큰 수, 문서 토큰 수)
    return float(sim.max(axis=1).sum())


class ColbertIndex:
    """RAGatouille의 RAG.index / RAG.search에 해당하는 간단한 인메모리 인덱스."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = LateInteractionTextEmbedding(model_name)
        self.chunks: list[str] = []
        self.doc_vecs: list[np.ndarray] = []

    def index(self, texts: list[str]):
        self.chunks = texts
        self.doc_vecs = list(self.model.embed(texts))  # 각 원소: (토큰 수, 128)

    def search(self, query: str, k: int = 3) -> list[dict]:
        query_vec = next(iter(self.model.query_embed(query)))
        scores = [maxsim(query_vec, d) for d in self.doc_vecs]
        top = np.argsort(scores)[::-1][:k]
        return [
            {"rank": r + 1, "score": scores[i], "content": self.chunks[i]}
            for r, i in enumerate(top)
        ]


class ColbertRetriever(BaseRetriever):
    """RAG.as_langchain_retriever()에 해당: LangChain 체인에 연결 가능."""

    colbert: ColbertIndex
    k: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(page_content=r["content"], metadata={"score": r["score"]})
            for r in self.colbert.search(query, self.k)
        ]


if __name__ == "__main__":
    full_document = get_wikipedia_page("Hayao_Miyazaki")

    # RAGatouille의 max_document_length=180(토큰)과 비슷하게 약 700자로 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_text(full_document)
    print(f"문서 길이: {len(full_document)}자 → {len(chunks)}개 청크")

    # 인덱스 생성 (처음 실행 시 모델 다운로드 약 0.44GB)
    colbert = ColbertIndex()
    colbert.index(chunks)
    print(f"첫 청크 벡터 모양: {colbert.doc_vecs[0].shape}  ← (토큰 수, 128)")

    # 쿼리
    query = "What animation studio did Miyazaki found?"
    print(f"\n[검색] {query}")
    for r in colbert.search(query, k=3):
        print(f"  {r['rank']}. (score {r['score']:.2f}) {r['content'][:120]}...")

    # 랭체인에 전달
    retriever = ColbertRetriever(colbert=colbert, k=3)
    docs = retriever.invoke(query)
    print(f"\n[LangChain retriever] {len(docs)}개 Document 반환")
