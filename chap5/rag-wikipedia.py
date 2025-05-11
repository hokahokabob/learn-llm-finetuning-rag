# -*- coding: sjis -*-
# -----------------------------------------------------
#  �f�[�^�x�[�X�̍\�z
# -----------------------------------------------------

# -----------------------------------------------------
#  ������̍\�z
# -----------------------------------------------------

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(lang="ja",
   # �ő咷�� 1K ��z�肵�āA�ȉ���2��ݒ�
   doc_content_chars_max=500, # 1����500�����ȉ�
   top_k_results=2 # 2������
)

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
)

# -----------------------------------------------------
#  �v�����v�g�̏���
# -----------------------------------------------------

template = """
### ���[�U:
�ȉ��̃e�L�X�g���Q�Ƃ��āA����ɑ�������ɓ����Ă��������B

{context}

{question}

### �V�X�e��:"""

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

q = "����u���т܂�q�����v�̌���҂͒N�ł����H"
ans = qa.invoke(q)

import re
pattern = re.compile(r'�V�X�e��:(.*)')
match = pattern.search(ans['result'])
ans0 = match.group(1)
print(ans0)




