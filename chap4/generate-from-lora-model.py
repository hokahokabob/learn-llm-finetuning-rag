# -*- coding: sjis -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "line-corporation/japanese-large-lm-3.6b"

base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)

lora_name = "output/checkpoint-3278"

model = PeftModel.from_pretrained(base_model, lora_name)


template = (
        "ユーザー:{instruction}\n"
        "システム:{output}"
)

q = "茨城県の観光名所を3つあげて下さい"
d = {'instruction':q, 'output':''}
ptext = template.format_map(d)

input_ids = tokenizer.encode(ptext, 
     add_special_tokens=False, 
     return_tensors="pt").to(device)

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

