# -*- coding: sjis -*-

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

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs = {'device':'cuda:0'},
)

from langchain_community.vectorstores import FAISS

db = FAISS.from_texts(texts, embeddings)

db.save_local('ibaraki.db')


        
