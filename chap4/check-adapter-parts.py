import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "line-corporation/japanese-large-lm-3.6b"
model = AutoModelForCausalLM.from_pretrained(model_name,
                         torch_dtype=torch.bfloat16)

import re

model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)
linear_layer_names = list(set(linear_layer_names))
print(linear_layer_names)
