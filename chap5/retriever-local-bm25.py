# -*- coding: sjis -*-

from langchain_community.retrievers import BM25Retriever
from janome.tokenizer import Tokenizer
import pickle

t = Tokenizer()

def my_preprocess_func(text):
    keywords = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if (pos in ["名詞", "動詞", "形容詞"]):
            keywords.append(token.surface)
    return keywords

with open('ibaraki-bm25.pkl', 'rb') as f:
    db = pickle.load(f)

q = "日立市のかみね動物園の開園時間"
docs = db.get_relevant_documents(q)

print(docs[0].page_content)

# retriever = db.as_retriever(search_kwargs={'k':2})
