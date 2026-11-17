from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

PYTHON_CODE = """
def hello_world():
  print("Hello, world!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
splitted_documents = python_splitter.split_text(PYTHON_CODE)
print(splitted_documents)
