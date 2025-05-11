# -*- coding: sjis -*-

#------------------------------------------
#  ���f���� tokenizer �̐ݒ�
#------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-small"
model = AutoModelForCausalLM.from_pretrained(model_name, 
                  torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#----------------------------------
#  �f�[�^�̃_�E�����[�h
#----------------------------------

import datasets
dolly_ja = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")

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

#------------------------------------------
#  �f�[�^�i�v�����v�g�j�̃��X�g�̍쐬
#------------------------------------------

datalist = []
for i in range(len(dolly_ja['train'])):
    d = dolly_ja['train'][i]
    if (d['input'] == ''):
        ptext = template['wo_input'].format_map(d)
    else:
        ptext = template['w_input'].format_map(d)
    if (len(ptext) < 1500):
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
    output_dir='./output',
    num_train_epochs=5,    
    save_steps=2000,
    per_device_train_batch_size=1
)

trainer = Trainer(
    model=model,
    data_collator=collator,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
