# -*- coding: sjis -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Saved model
model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-88000/")

# Original  model 
# model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-small")

tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small")

input = tokenizer.encode("ほな言うのもなんやけど、", return_tensors="pt")

with torch.no_grad():
    tokens = model.generate(input,max_new_tokens=40,do_sample=True)
output = tokenizer.decode(tokens[0], skip_special_tokens=True)

print(output)
