# -*- coding: sjis -*-

# -----------------------------------------------------
#  データベースの読み込み
# -----------------------------------------------------

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device':'cpu'},
#    encode_kwargs = {'normalize_embeddings': False}
)

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# 保存してあるデータベースの読み込み
db = FAISS.load_local('joseito.db',embeddings, allow_dangerous_deserialization=True)

# -----------------------------------------------------
#  検索器の構築
# -----------------------------------------------------

retriever = db.as_retriever()  # 検索文書数 4 

# 検索文書数 k = 2 の場合
# retriever = db.as_retriever(search_kwargs={'k':2})

# -----------------------------------------------------
#  OpenAI の API キーの設定
# -----------------------------------------------------

import os

os.environ['OPENAI_API_KEY'] = 'sk-*****'

# -----------------------------------------------------
#  RAG の構築
# -----------------------------------------------------

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever,
    return_source_documents=True,
)

# -----------------------------------------------------
# 実行例
# -----------------------------------------------------

q = "太宰治の小説である「女生徒」の主人公の一番好きな子の名前は何ですか？"
ans = qa.invoke(q)

print(ans['result'])
# --> 太宰治の小説「女生徒」の主人公が一番好きな子の名前は、新ちゃんです。

docs = ans['source_documents']
for d in docs:
    print()
    print(d.page_content[:100])

# -->     
# 太宰治 女生徒\n女生徒\n太宰治
# 
# おやすみなさい。私は、王子さまの・・・文庫、角川書店
# 
# しの一ばんの親友です、なんて皆に・・・で、私も、さすが
# 
# 二さんの弟で、私とは同じとしなん・・・一ばん新ちゃんを好きだ・・・なんという

