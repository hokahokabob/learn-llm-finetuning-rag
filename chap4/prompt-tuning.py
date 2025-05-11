# -*- coding: sjis -*-

#----------------------------------
# Prompt Tuning �̐ݒ�
#----------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType

model_name = "line-corporation/japanese-large-lm-3.6b"

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, legacy=True)

pt_config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    token_dim=768,
    prompt_tuning_init_text="���[�U�[�̎w���⎿��ɓ����ĉ�����",
    tokenizer_name_or_path=model_name
)

#------------------------------------------
#  Prompt Tuning  ���f���̐ݒ�
#------------------------------------------

model = get_peft_model(base_model, pt_config)

#--------------------------------------------
# �ȉ��� Instruction Turinig �ƑS������

#----------------------------------
#  �f�[�^�̃_�E�����[�h
#----------------------------------

import datasets
dolly_ja = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")

#----------------------------------
#  �e���v���[�g
#----------------------------------

template = (
        "���[�U�[:{instruction}\n"
        "�V�X�e��:{output}"
)

#------------------------------------------
#  �f�[�^�i�v�����v�g�j�̃��X�g�̍쐬
#------------------------------------------

datalist = []
for i in range(len(dolly_ja['train'])):
    d = dolly_ja['train'][i]
    if (d['input'] == ''):
        ptext = template.format_map(d)
        if (len(ptext) < 100):
            datalist.append(ptext)

#------------------------------------------
#  train_dataset �̍\�z
#------------------------------------------

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, datalist, tokenizer):
        self.tokenizer = tokenizer
        self.features = []
        for ptext in datalist:
            input_ids = self.tokenizer.encode(ptext)
            input_ids = input_ids + [ self.tokenizer.eos_token_id ]
            input_ids = torch.LongTensor(input_ids)
            self.features.append({'input_ids': input_ids})
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

train_dataset = MyDataset(datalist, tokenizer)

#------------------------------------------
#  Trainer �̐ݒ�Ɗw�K�̎��s
#------------------------------------------

from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./output-pt',
    num_train_epochs=1,
    save_strategy='epoch',    
    per_device_train_batch_size=1,
    logging_steps=20,
)

trainer = Trainer(
    model=model,
    data_collator=collator,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
