# -*- coding: sjis -*-

# ------------------------------
# モデルの設定
# ------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ------------------------------
# Dataset の設定
# ------------------------------

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, filename, tokenizer):
        self.tokenizer = tokenizer
        self.features = []
        with open(filename,'r') as f:
            lines =  f.read().split('\n')
            for line in lines:
                input_ids = self.tokenizer.encode(line,
                        padding='longest',
                        max_length=512,
                        return_tensors='pt')[0]
                self.features.append({'input_ids': input_ids})
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]

train_dataset = MyDataset('train.txt', tokenizer)

# ------------------------------
# dataloader の設定
# ------------------------------

from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

from torch.utils.data import DataLoader

dataloader = DataLoader(train_dataset, batch_size=10,
                        shuffle=True, collate_fn=collator)

# ------------------------------
# Trainer の設定
# ------------------------------

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=10,
    per_device_train_batch_size=10,
)

trainer = Trainer(
    model=model,
    data_collator=collator,
    args=training_args,
    train_dataset=train_dataset
)

# ------------------------------
# 学習の実行
# ------------------------------

trainer.train()
