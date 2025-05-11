from langchain_community.document_loaders import UnstructuredURLLoader
urls = ["https://www.aozora.gr.jp/cards/000035/files/275_13903.html"]
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()

from langchain_community.document_transformers import Html2TextTransformer

html2text = Html2TextTransformer()
plain_text = html2text.transform_documents(docs)

print(plain_text[0].page_content[0:50])
