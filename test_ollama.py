"""LangChain + Ollama 연결 확인 스크립트.

실행 전:
    conda activate langchain

실행:
    python test_ollama.py
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate

model = "qwen2.5:14b"


def test_chat():
    chat = ChatOllama(model=model)
    messages = [
        SystemMessage(content="문장 끝에 느낌표를 3개 붙여서 한국어로 답해."),
        HumanMessage(content="한국의 수도는 어디입니까?"),
    ]
    resp = chat.invoke(messages)
    print("응답:", resp.content)


if __name__ == "__main__":
    try:
        test_chat()
    except Exception as e:
        print(f"\n채팅 테스트 실패: {e}")
        print("  → Ollama 실행 여부(ollama list)와 모델 이름을 확인하세요.")
        raise SystemExit(1)

    print("\n완료: LangChain + Ollama 정상 동작")
