from langchain_community.retrievers import BM25Retriever

texts = [
   "foo", "bar", "world", "hello", "foo bar"
]

retriever = BM25Retriever.from_texts(texts)
result = retriever.get_relevant_documents("foo")
print(result)
# -->
# [Document(page_content='foo'),
#  Document(page_content='foo bar'),
#  Document(page_content='hello'),
#  Document(page_content='world')]
