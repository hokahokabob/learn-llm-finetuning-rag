# -*- coding: sjis -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)

# --------------------------------------------------------------------
# >>> input = tokenizer.encode("私は犬が好き。", return_tensors="pt")
# >>> print(input)
# tensor([[2727, 3807, 9439,  247]])
# >>> a = [tokenizer.decode(input[0][i]) for i in range(len(input[0]))]
# >>> print(a)
# ['私は', '犬', 'が好き', '。']
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# >>> output = model(input)
# >>> type(output)
# <class 'transformers.modeling_outputs.CausalLMOutputWithPast'>
# >>> print(output.logits)
# tensor([[[  7.2745, -17.2270,   8.0928,  ..., -16.3880, -17.2634, -17.3245],
#          [  7.2613, -16.8069,  10.7188,  ..., -16.0717, -17.0571, -16.9093],
#          [ 12.8202, -16.5408,  15.7864,  ..., -15.9593, -16.8466, -16.8622],
#          [ 12.0015, -17.0628,   7.1439,  ..., -16.2650, -17.1278, -17.2398]]],
#        grad_fn=<UnsafeViewBackward0>)
# >>> print(output.logits.shape)
# torch.Size([1, 4, 52096])
# --------------------------------------------------------------------

loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(output.logits[0], torch.tensor([3807, 9439, 247, -100]))

# --------------------------------------------------------------------
# >>> loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
# >>> loss0 = loss_fn(output.logits[0], torch.tensor([3807, 9439, 247, -100]))
# >>> print(loss0)
# tensor([8.0847, 5.4860, 3.0050, 0.0000], grad_fn=<NllLossBackward0>)
# >>> print(torch.sum(loss0)/3)
# tensor(5.5252, grad_fn=<DivBackward0>)
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# >>> loss_fn = torch.nn.CrossEntropyLoss()
# >>> loss1 = loss_fn(output.logits[0], torch.tensor([3807, 9439, 247, -100]))
# >>> print(loss1)
# tensor(5.5252, grad_fn=<NllLossBackward0>)
# --------------------------------------------------------------------

optimizer.zero_grad()
loss.backward()
optimizer.step()

