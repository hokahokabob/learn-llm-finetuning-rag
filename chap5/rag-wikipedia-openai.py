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

q = "����u���т܂�q�����v�̌���҂͒N�ł����H"
ans = qa.invoke(q)
print(ans["result"])
