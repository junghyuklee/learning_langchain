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
query = "일어나서 이를 닦고 뉴스를 읽었어요. 그러다 전자레인지에 음식을 넣어둔 걸 깜빡했네요. 인공지능의 발전으로 인한 부작용이 뭐가 있어?"
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
    return answer.content


result = qa.invoke(query)
print(result)
