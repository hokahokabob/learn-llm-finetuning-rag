# -*- coding: sjis -*-

from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os

os.environ['OPENAI_API_KEY'] = 'sk-*****'

template = """����ɉ񓚂��ĉ������B
����F{question}
�񓚁F"""

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

q = "����w�H�w���͂ǂ��ɂ���܂����H"

hyde_vec = hyde_embd.embed_query(q)

print(type(hyde_vec))
# --> <class 'list'>
print(len(hyde_vec))
# --> 1536


