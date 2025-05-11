# -*- coding: sjis -*-

with open('joseito.txt','r',encoding='utf-8') as f:
    text = f.read()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,   # �`�����N�̕�����
    chunk_overlap  = 0,  # �`�����N�I�[�o�[���b�v�̕�����
)

texts = text_splitter.split_text(text)

# ------------------------------------------
## �s�d�w�s�r �̃^�C�v���m�F
# print(type(texts))
# --> <class 'list'>
# ------------------------------------------
## �`�����N�̑����ƒ��g�̊m�F
#  print(len(texts))
# --> 367
#  print(texts[0])
# --> ���Ɏ� �����k\n�����k\n���Ɏ�
#  print(texts[1])
# --> '�����A������܂��Ƃ��̋C���́A�E�E�E�A����A������'

from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
#    model_kwargs = {'device':'cuda:0'},
#    encode_kwargs = {'normalize_embeddings': False}
)
# ------------------------------------------
# d0 = "���͌����D���B"
# d1 = "�ނ̌��͂���������B"
# a = embeddings.embed_documents([d0, d1])
# print(type(a[0]))
# --> <class 'list'>
# print(len(a[0]))
# --> 1024

from langchain_community.vectorstores import FAISS

db = FAISS.from_texts(texts, embeddings)

# �f�[�^�x�[�X�̕ۑ�

db.save_local('joseito.db')

# �ۑ������f�[�^�x�[�X�̓ǂݍ���
# db = FAISS.load_local('joseito.db',embeddings, allow_dangerous_deserialization=True)

# ------------------------------------------
# a = db.similarity_search("���͌����D���B")
# print(len(a))
# --> 4
# print(type(a[0]))
# --> <class 'langchain_core.documents.base.Document'>
# print(a[0].page_content)
# --> ���������������Ǝv���܂��B
# e = embeddings.embed_documents(["���͌����D���B"])
# b = db.similarity_search_by_vector(e[0])
# print(b[0].page_content)
# --> ���������������Ǝv���܂��B

