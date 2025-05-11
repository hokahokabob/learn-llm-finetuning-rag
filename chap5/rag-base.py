# -*- coding: sjis -*-
# -----------------------------------------------------
#  �f�[�^�x�[�X�̍\�z
# -----------------------------------------------------

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
    model_kwargs = {'device':'cuda:0'},
#    encode_kwargs = {'normalize_embeddings': False}
)

from langchain_community.vectorstores import FAISS

# �ۑ����Ă���f�[�^�x�[�X�̓ǂݍ���
db = FAISS.load_local('joseito.db',embeddings, allow_dangerous_deserialization=True)

# -----------------------------------------------------
#  ������̍\�z
# -----------------------------------------------------

# retriever = db.as_retriever()
retriever = db.as_retriever(search_kwargs={'k':2})

# -----------------------------------------------------
#  ���f���̏���
# -----------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "line-corporation/japanese-large-lm-3.6b-instruction-sft"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
#    device_map="cuda:0",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.01,
    repetition_penalty=2.0,
)

# -----------------------------------------------------
#  �v�����v�g�̏���
# -----------------------------------------------------

template = """
���[�U�[:�ȉ��̃e�L�X�g���Q�Ƃ��āA����ɑ�������ɓ����Ă��������B

{context}

{question}

�V�X�e��:"""

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string"
)

# -----------------------------------------------------
#  RetrievalQA �̃C���X�^���X�쐬
# -----------------------------------------------------

from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

qa = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=pipe),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
    verbose=True,
)

# -----------------------------------------------------
#  ���s��
# -----------------------------------------------------

q = "��l���̈�ԍD���Ȏq�̖��O�͉��ł����H"
ans = qa.invoke(q)
print(ans['result'])

print("--------------------------")

import re
pattern = re.compile(r'�V�X�e��:(.*)',re.DOTALL)
match = pattern.search(ans['result'])
ans0 = match.group(1)
print(ans0)

# docs = ans['source_documents']
# for d in docs:
#     print()
#     print(d.page_content[:100])
