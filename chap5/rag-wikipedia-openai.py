# -*- coding: sjis -*-

import os

os.environ['OPENAI_API_KEY'] = 'sk-*****'

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(lang="ja",
    doc_content_chars_max=500,
    top_k_results=2
)

llm = ChatOpenAI()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

q = "漫画「ちびまる子ちゃん」の原作者は誰ですか？"
ans = qa.invoke(q)
print(ans["result"])
