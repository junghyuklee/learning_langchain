"""LangChain + Ollama 연결 확인 스크립트.

실행 전:
    conda activate langchain

실행:
    python test_ollama.py
"""

from langchain_ollama import ChatOllama

CHAT_MODEL = "llama3.2"


def test_chat():
    print(f"[1] 채팅 테스트 (model={CHAT_MODEL})")
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)
    resp = llm.invoke("LangChain이 뭔지 한 문장으로 설명해줘")
    print("  응답:", resp.content)


if __name__ == "__main__":
    try:
        test_chat()
    except Exception as e:
        print(f"\n채팅 테스트 실패: {e}")
        print("  → Ollama 실행 여부(ollama list)와 모델 이름을 확인하세요.")
        raise SystemExit(1)

    print("\n완료: LangChain + Ollama 정상 동작")
