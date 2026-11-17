from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class AnswerWithJustification(BaseModel):
    """사용자의 질문에 대한 답변과 그에 대한 근거(justification)를 함께 제공하세요."""

    answer: str = Field(description="사용자의 질문에 대한 답변")
    justification: str = Field(description="답변에 대한 근거")


llm = ChatOllama(model="qwen2.5:14b", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)

result = structured_llm.invoke(
    "1 킬로그램의 벽돌과 1 킬로그램의 솜 중 어느 것이 더 무겁나요?"
)

print(result.model_dump_json())
