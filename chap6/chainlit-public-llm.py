# -*- coding: sjis -*-

import chainlit as cl

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "line-corporation/japanese-large-lm-3.6b-instruction-sft"

model = AutoModelForCausalLM.from_pretrained(model_name,
                        device_map="auto", 
                        torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, use_fast=False)

template = (
        "ユーザー:{instruction}\n"
        "システム:{output}"
)

def llm_main(q):
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
    ans = tokenizer.decode(tokens[0][start_pos:], skip_special_tokens=True)
    return ans

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="何か入力せよ").send()

@cl.on_message
async def on_message(input_message):
    ans = llm_main(input_message.content)
    await cl.Message(content=ans).send()
