from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant."), ("user", "{input}")]
)

model = ChatOllama(model="qwen2.5:14b", temperature=0)


@chain
def chatbot(values):
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token


for part in chatbot.stream(
    {"1 킬로그램의 벽돌과 1 킬로그램의 솜 중 어느 것이 더 무겁나요?"}
):
    print(part)
