# -*- coding: sjis -*-

# -----------------------------------------------------
#  データベースの構築
# -----------------------------------------------------

from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name = "intfloat/multilingual-e5-large",
#    model_kwargs = {'device':'cuda:0'},
#    encode_kwargs = {'normalize_embeddings': False}
)

from langchain_community.vectorstores import FAISS

# 保存してあるデータベースの読み込み

db = FAISS.load_local('ibaraki.db',embeddings, allow_dangerous_deserialization=True)

# -----------------------------------------------------
#  検索器の構築
# -----------------------------------------------------

retriever = db.as_retriever()

# -----------------------------------------------------
#  モデルの準備
# -----------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
).eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.01,
)

# -----------------------------------------------------
#  プロンプトの準備
# -----------------------------------------------------

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "以下の文章を参照することで、ユーザーからの質問にできるだけ正確に答えてください。"
text = "{context}\nユーザーからの質問は次のとおりです。{question}"

template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)


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

q = "現在の日立駅の駅舎は誰の作品ですか？"
ans = qa.invoke(q)
# print(ans['result'])

import re
pattern = re.compile(r'\[/INST\](.*)', re.DOTALL)
match = pattern.search(ans['result'])
ans0 = match.group(1)
print(ans0)



