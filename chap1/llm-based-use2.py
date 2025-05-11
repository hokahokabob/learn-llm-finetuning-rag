# -*- coding: sjis -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "cyberagent/open-calm-small"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline("text-generation", 
                     model=model, 
                     tokenizer=tokenizer)

outs = generator("東京は日本の", max_new_tokens=30)
print(outs[0])

