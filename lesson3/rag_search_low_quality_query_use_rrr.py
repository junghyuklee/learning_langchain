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

rewrite_prompt = ChatPromptTemplate.from_template(
    "다음 입력에서 질문과 관계없는 내용은 버리고, 핵심 질문만 한 문장으로 다시 쓰세요.\n"
    "설명이나 따옴표 없이, 다시 쓴 질문 한 문장만 출력하세요.\n\n입력: {x}"
)
query = "일어나서 이를 닦고 뉴스를 읽었어요. 그러다 전자레인지에 음식을 넣어둔 걸 깜빡했네요. 인공지능의 발전으로 인한 부작용이 뭐가 있어?"
# 내부 검색 데이터 활용을 위해 창의성을 배제하도록 temperature 를 0으로 설정
llm = ChatOllama(model="qwen2.5:14b", temperature=0)


def parse_rewriter_output(message):
    return message.content.strip("'").strip("**")


rewriter = rewrite_prompt | llm | parse_rewriter_output


@chain
def qa(input):
    # 쿼리 재작성
    new_query = rewriter.invoke(input)
    print("재작성한 쿼리: ", new_query)
    # 관련 문서 검색
    docs = retriever.invoke(new_query)
    # 원문만 이어붙이는게 깔끔
    context = "\n\n".join(d.page_content for d in docs)
    # 답변 생성
    answer = (prompt | llm).invoke({"context": context, "question": input})
    return answer.content


result = qa.invoke(query)
print(result)
