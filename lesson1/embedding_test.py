from langchain_ollama import OllamaEmbeddings

model = OllamaEmbeddings(model="bge-m3")
embeddings = model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World",
    ]
)

print(len(embeddings), "개 문장")
print(len(embeddings[0]), "차원")
print(embeddings)
