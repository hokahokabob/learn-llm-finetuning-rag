# -*- coding: sjis -*-

from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os

os.environ['OPENAI_API_KEY'] = 'sk-*****'

template = """質問に回答して下さい。
質問：{question}
回答："""

llm = ChatOpenAI()

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
#    model_kwargs = {'device':'cuda:0'},
#    encode_kwargs = {'normalize_embeddings': False}
)

hyde_embd = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embeddings,
)

q = "茨城大学工学部はどこにありますか？"

hyde_vec = hyde_embd.embed_query(q)

# ここまでが rag-hyde-openai.py
#-------------------------------------

from langchain_community.vectorstores import FAISS

# 保存してあるデータベースの読み込み
db = FAISS.load_local('ibaraki.db',embeddings, allow_dangerous_deserialization=True)

# -----------------------------------------------------
#  検索
# -----------------------------------------------------

docs = db.similarity_search_by_vector(hyde_vec, k=2)

# -----------------------------------------------------
#  プロンプトの作成
# -----------------------------------------------------

template2 = """
ユーザー:以下のテキストを参照して、それに続く質問に答えてください。

{context1}

{context2}

{question}

システム:"""

d = {}
d['context1'] = docs[0].page_content
d['context2'] = docs[1].page_content
d['question'] = q

prompt = PromptTemplate(
    template=template2,
    input_variables=["context1", "context2", "question"],
    template_format="f-string"
)

# -----------------------------------------------------
#  OpenAI のモデルを使って回答
# -----------------------------------------------------

chain = LLMChain(llm=llm,prompt=prompt)
ans = chain.invoke(d)
print(ans['text'])

