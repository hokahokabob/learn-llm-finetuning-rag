# -*- coding: sjis -*-

#------------------------------------------
#  ���f���� tokenizer �̐ݒ�
#------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/checkpoint-72000/")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small")

#----------------------------------
#  �e���v���[�g
#----------------------------------

template = {
    "w_input": (
        "�ȉ��̓^�X�N���L�q�����w���Ɠ��͂ł��B���͂̓^�X�N�ŎQ�Ƃ���镶�͂ł��B�w����K�؂ɖ����������������Ȃ����B\n\n"
        "### �w��:\n{instruction}\n\n"
        "### ����:\n{input}\n\n"
        "### ����:\n{output}"
    ),
    "wo_input": (
        "�ȉ��̓^�X�N���L�q�����w���ł��B�v����K�؂ɖ����������������Ȃ����B\n\n"
        "### �w��:\n{instruction}\n\n"
        "### ����:\n{output}"
    )
}

#----------------------------------
#  ���s��
#----------------------------------

d = {}
d['instruction'] = "���{�ň�ԍ����R�͉��ł����H"
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



















