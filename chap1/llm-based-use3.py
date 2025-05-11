# -*- coding: sjis -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "cyberagent/open-calm-small"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

input = tokenizer("東京は日本の", return_tensors="pt")
tokens = model.generate(**input,max_new_tokens=30)

output = tokenizer.decode(tokens[0], skip_special_tokens=True) 
print(output)
