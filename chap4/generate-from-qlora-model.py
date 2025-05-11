# -*- coding: sjis -*-

#----------------------------------
# �ʎq���̐ݒ�
#----------------------------------

import torch
from transformers import BitsAndBytesConfig
from peft import PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#------------------------------------------
#  �ʎq���������f���� tokenizer �̐ݒ�
#------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-7b"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

#----------------------------------------

qlora_name = "output-qlora/checkpoint-3278"

model = PeftModel.from_pretrained(base_model, qlora_name)

template = (
        "���[�U�[:{instruction}\n"
        "�V�X�e��:{output}"
)

q = "�j�Z�����d�ƌ��q�͔��d�̈Ⴂ��������ĉ�����"
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
