from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
items = parser.invoke("사과, 바나나, 체리")

print(items)  # ['사과', '바나나', '체리']
