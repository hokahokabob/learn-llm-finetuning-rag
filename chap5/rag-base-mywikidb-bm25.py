# -*- coding: sjis -*-
# -----------------------------------------------------
#  データベースの構築
# -----------------------------------------------------

from langchain_community.retrievers import BM25Retriever
from janome.tokenizer import Tokenizer
import pickle

t = Tokenizer()

def my_preprocess_func(text):
    keywords = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if (pos in ["名詞", "動詞", "形容詞"]):
            keywords.append(token.surface)
    return keywords

with open('ibaraki-bm25.pkl', 'rb') as f:
    retriever = pickle.load(f)

# -----------------------------------------------------
#  モデルの準備
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
#  プロンプトの準備
# -----------------------------------------------------

template = """
### ユーザ:
以下のテキストを参照して、それに続く質問に答えてください。

{context}

{question}

### システム:"""

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
    template_format="f-string"
)

# -----------------------------------------------------
#  RetrievalQA のインスタンス作成
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
#  実行例
# -----------------------------------------------------

q = "茨城大学の本部はどこにありますか？"
ans = qa.invoke(q)
# print(ans['result'])

import re
pattern = re.compile(r'システム:(.*)',re.DOTALL)
match = pattern.search(ans['result'])
ans0 = match.group(1)
print(ans0)
# --> 茨城大学は、水戸市文京二丁目1-1に本部を置いています。
