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

# �����܂ł� rag-hyde-openai.py
#-------------------------------------

from langchain_community.vectorstores import FAISS

# �ۑ����Ă���f�[�^�x�[�X�̓ǂݍ���
db = FAISS.load_local('ibaraki.db',embeddings, allow_dangerous_deserialization=True)

# -----------------------------------------------------
#  ����
# -----------------------------------------------------

docs = db.similarity_search_by_vector(hyde_vec, k=2)

# -----------------------------------------------------
#  �v�����v�g�̍쐬
# -----------------------------------------------------

template2 = """
���[�U�[:�ȉ��̃e�L�X�g���Q�Ƃ��āA����ɑ�������ɓ����Ă��������B

{context1}

{context2}

{question}

�V�X�e��:"""

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
#  OpenAI �̃��f�����g���ĉ�
# -----------------------------------------------------

chain = LLMChain(llm=llm,prompt=prompt)
ans = chain.invoke(d)
print(ans['text'])

