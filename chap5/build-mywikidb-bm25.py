# -*- coding: sjis -*-

from janome.tokenizer import Tokenizer

t = Tokenizer()

def my_preprocess_func(text):
    keywords = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if (pos in ["名詞", "動詞", "形容詞"]):
            keywords.append(token.surface)
    return keywords

from datasets import load_dataset

wikija_dataset = load_dataset(
    path="singletongue/wikipedia-utils",
    name="passages-c400-jawiki-20230403",
    split="train",
)

ibaraki = ""
tstr = '茨城'
for data in wikija_dataset:
    if ((tstr in data['title']) or (tstr in data['text'])):
        ibaraki += (data['text'] + "\n")

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,      # チャンクの文字数
    chunk_overlap  = 100,  # チャンクオーバーラップの文字数
)

texts = text_splitter.split_text(ibaraki)

from langchain_community.retrievers import BM25Retriever

db = BM25Retriever.from_texts(
          texts,
          preprocess_func=my_preprocess_func,
)

import pickle
with open('ibaraki-bm25.pkl','wb') as f:
    pickle.dump(db, f)
