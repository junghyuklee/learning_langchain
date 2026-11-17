import os

# WebBaseLoader import 전에 설정해야 경고가 사라짐
os.environ.setdefault("USER_AGENT", "learning-langchain/0.1")

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.langchain.com")
documents = loader.load()

print(documents)
