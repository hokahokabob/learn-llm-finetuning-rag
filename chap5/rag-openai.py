# -*- coding: sjis -*-

# -----------------------------------------------------
#  �f�[�^�x�[�X�̓ǂݍ���
# -----------------------------------------------------

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={'device':'cpu'},
#    encode_kwargs = {'normalize_embeddings': False}
)

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# �ۑ����Ă���f�[�^�x�[�X�̓ǂݍ���
db = FAISS.load_local('joseito.db',embeddings, allow_dangerous_deserialization=True)

# -----------------------------------------------------
#  ������̍\�z
# -----------------------------------------------------

retriever = db.as_retriever()  # ���������� 4 

# ���������� k = 2 �̏ꍇ
# retriever = db.as_retriever(search_kwargs={'k':2})

# -----------------------------------------------------
#  OpenAI �� API �L�[�̐ݒ�
# -----------------------------------------------------

import os

os.environ['OPENAI_API_KEY'] = 'sk-*****'

# -----------------------------------------------------
#  RAG �̍\�z
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
# ���s��
# -----------------------------------------------------

q = "���Ɏ��̏����ł���u�����k�v�̎�l���̈�ԍD���Ȏq�̖��O�͉��ł����H"
ans = qa.invoke(q)

print(ans['result'])
# --> ���Ɏ��̏����u�����k�v�̎�l������ԍD���Ȏq�̖��O�́A�V�����ł��B

docs = ans['source_documents']
for d in docs:
    print()
    print(d.page_content[:100])

# -->     
# ���Ɏ� �����k\n�����k\n���Ɏ�
# 
# ���₷�݂Ȃ����B���́A���q���܂́E�E�E���ɁA�p�쏑�X
# 
# ���̈�΂�̐e�F�ł��A�Ȃ�ĊF�ɁE�E�E�ŁA�����A������
# 
# �񂳂�̒�ŁA���Ƃ͓����Ƃ��Ȃ�E�E�E��΂�V�������D�����E�E�E�Ȃ�Ƃ���

