# -*- coding: sjis -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "line-corporation/japanese-large-lm-3.6b"

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)

pt_name = "output-pt/checkpoint-3278"

model = PeftModel.from_pretrained(base_model, pt_name).to(device)

template = (
        "ユーザー:{instruction}\n"
        "システム:{output}"
)

d = {}
d['instruction'] = "夕日が赤い理由を教えて下さい"
d['output'] = ''

ptext = template.format_map(d)

input_ids = tokenizer.encode(ptext, add_special_tokens=False, return_tensors="pt").to(device)
start_pos = len(input_ids[0])

with torch.no_grad():
    tokens = model.generate(input_ids=input_ids,
                            max_new_tokens=200,
                            temperature=1.0,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,                            
    )

output = tokenizer.decode(tokens[0][start_pos:], skip_special_tokens=True)
print(output)
