# -*- coding: sjis -*-

#------------------------------------------
#  モデルと tokenizer の設定
#------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-72000/")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small")

#----------------------------------
#  テンプレート
#----------------------------------

template = {
    "w_input": (
        "以下はタスクを記述した指示と入力です。入力はタスクで参照される文章です。指示を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n"
        "### 入力:\n{input}\n\n"
        "### 応答:\n{output}"
    ),
    "wo_input": (
        "以下はタスクを記述した指示です。要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n"
        "### 応答:\n{output}"
    )
}

#----------------------------------
#  実行例
#----------------------------------

d = {}
d['instruction'] = "日本で一番高い山は何ですか？"
d['output'] = ''

ptext = template['wo_input'].format_map(d)

input = tokenizer.encode(ptext, return_tensors="pt")
start_pos = len(input[0])

with torch.no_grad():
    tokens = model.generate(input,max_new_tokens=60,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
#                            temperature=0.0)
    
output = tokenizer.decode(tokens[0][start_pos:], skip_special_tokens=True)
print(output)



















