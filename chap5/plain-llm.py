# -*- coding: sjis -*-

# -----------------------------------------------------
#  ���f���̏���
# -----------------------------------------------------

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "line-corporation/japanese-large-lm-3.6b-instruction-sft"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False, use_fast=False)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).eval()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.01,
)

# -----------------------------------------------------
#  �v�����v�g�̏���
# -----------------------------------------------------

template = """
���[�U�[:{question}

�V�X�e��:"""

# -----------------------------------------------------
#  ���s��
# -----------------------------------------------------

pattern = re.compile(r'�V�X�e��:(.*)')

q = {}
q['question'] = "����u�h���S���{�[���v�̌���҂͒N�ł����H"
input = template.format_map(q)
ans = pipe(input)
print(ans[0]['generated_text'])

print("-"*20)

q = {}
q['question'] = "����u���т܂�q�����v�̌���҂͒N�ł����H"
input = template.format_map(q)
ans = pipe(input)
print(ans[0]['generated_text'])

