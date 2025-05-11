# -*- coding: sjis -*-

with open('joseito.txt','r',encoding='utf-8') as f:
    text = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,   # チャンクの文字数
    chunk_overlap  = 0,  # チャンクオーバーラップの文字数
)

texts = text_splitter.split_text(text)

# ------------------------------------------
## ＴＥＸＴＳ のタイプを確認
# print(type(texts))
# --> <class 'list'>
# ------------------------------------------
## チャンクの総数と中身の確認
#  print(len(texts))
# --> 367
#  print(texts[0])
# --> 太宰治 女生徒\n女生徒\n太宰治
#  print(texts[1])
# --> 'あさ、眼をさますときの気持は、・・・、いや、ちがう'

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
#    model_kwargs = {'device':'cuda:0'},
#    encode_kwargs = {'normalize_embeddings': False}
)
# ------------------------------------------
# d0 = "私は犬が好き。"
# d1 = "彼の犬はお利口さん。"
# a = embeddings.embed_documents([d0, d1])
# print(type(a[0]))
# --> <class 'list'>
# print(len(a[0]))
# --> 1024

from langchain_community.vectorstores import FAISS

db = FAISS.from_texts(texts, embeddings)

# データベースの保存

db.save_local('joseito.db')

# 保存したデータベースの読み込み
# db = FAISS.load_local('joseito.db',embeddings, allow_dangerous_deserialization=True)

# ------------------------------------------
# a = db.similarity_search("私は犬が好き。")
# print(len(a))
# --> 4
# print(type(a[0]))
# --> <class 'langchain_core.documents.base.Document'>
# print(a[0].page_content)
# --> 美しく生きたいと思います。
# e = embeddings.embed_documents(["私は犬が好き。"])
# b = db.similarity_search_by_vector(e[0])
# print(b[0].page_content)
# --> 美しく生きたいと思います。

