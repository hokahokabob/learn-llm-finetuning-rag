# -*- coding: sjis -*-

#------------------------------------------
#  モデルと tokenizer の設定
#------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-small"
model = AutoModelForCausalLM.from_pretrained(model_name, 
                  torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#----------------------------------
#  データのダウンロード
#----------------------------------

import datasets
dolly_ja = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")

#----------------------------------
#  テンプレート
#----------------------------------

template = {
    "w_input": (
        "以下はタスクを記述した指示と入力です。入力はタスクで参照される文章です。指示を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n"
        "### 入力:\n{input}\n\n"
        "### 応答:\n{output}"
    ),
    "wo_input": (
        "以下はタスクを記述した指示です。要求を適切に満たす応答を書きなさい。\n\n"
        "### 指示:\n{instruction}\n\n"
        "### 応答:\n{output}"
    )
}

#------------------------------------------
#  データ（プロンプト）のリストの作成
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
#  train_dataset の構築
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
#  Trainer の設定と学習の実行
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
