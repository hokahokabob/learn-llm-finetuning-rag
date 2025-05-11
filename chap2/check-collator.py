# -*- coding: sjis -*-

# --------------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --------------------------------------------------------------------

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

from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

from torch.utils.data import DataLoader

dataloader = DataLoader(train_dataset, batch_size=5,
                        shuffle=True, collate_fn=collator)

dl = dataloader.__iter__()
a = dl.__next__()
print(a)

# --------------------------------------------------------------------
# >>> out = model(**a)
# >>> print(out.loss)
# tensor(5.5240, grad_fn=<NllLossBackward0>)
# --------------------------------------------------------------------
