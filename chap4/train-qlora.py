# -*- coding: sjis -*-

#----------------------------------
# �ʎq���̐ݒ�
#----------------------------------

import torch
from transformers import BitsAndBytesConfig

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

#----------------------------------
# LoRA �̐ݒ�
#----------------------------------

from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)

#--------------------------------------------
# �ȉ��� Instruction Turinig �Ɗ�{�I�ɓ���

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
    output_dir='./output-qlora',
    num_train_epochs=1,
    save_strategy='epoch',  # epoch ���̕ۑ��ɕύX
    per_device_train_batch_size=1,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    data_collator=collator,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
