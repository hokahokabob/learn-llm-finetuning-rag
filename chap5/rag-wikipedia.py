# -*- coding: sjis -*-
# -----------------------------------------------------
#  データベースの構築
# -----------------------------------------------------

# -----------------------------------------------------
#  検索器の構築
# -----------------------------------------------------

from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(lang="ja",
   # 最大長が 1K を想定して、以下の2つを設定
   doc_content_chars_max=500, # 1文書500文字以下
   top_k_results=2 # 2件検索
)

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

q = "漫画「ちびまる子ちゃん」の原作者は誰ですか？"
ans = qa.invoke(q)

import re
pattern = re.compile(r'システム:(.*)')
match = pattern.search(ans['result'])
ans0 = match.group(1)
print(ans0)




