# -*- coding: sjis -*-

# -----------------------------------------------------
#  モデルの準備
# -----------------------------------------------------

import re
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
ユーザー:{question}

システム:"""

# -----------------------------------------------------
#  実行例
# -----------------------------------------------------

pattern = re.compile(r'システム:(.*)')

q = {}
q['question'] = "漫画「ドラゴンボール」の原作者は誰ですか？"
input = template.format_map(q)
ans = pipe(input)
print(ans[0]['generated_text'])

print("-"*20)

q = {}
q['question'] = "漫画「ちびまる子ちゃん」の原作者は誰ですか？"
input = template.format_map(q)
ans = pipe(input)
print(ans[0]['generated_text'])

