# -*- coding: sjis -*-

# [memo]
# from importlib import import_module
# llm = import_module('chap1.llm-based-use')
# input = llm.tokenizer("�����͓��{��", return_tensors="pt")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-small")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-small")

# --------------------------------------------------------------------
# >>> input = tokenizer("�����͓��{��", return_tensors="pt")
# >>> tokens = model.generate(**input, max_new_tokens=1,do_sample=False)
# ... Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
# >>> tokenizer.decode(tokens[0][-1])
# '��s'
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# >>> out = model.generate(**input, 
#                 max_new_tokens=1, 
#                 return_dict_in_generate=True, 
#                 output_scores=True)
# ... ... ... Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
# >>> out.scores[0].shape
# torch.Size([1, 52096])
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# >>> top5 = torch.topk(out.scores[0][0],5)
# >>> for i in range(5):
#     print(i+1,
#           tokenizer.decode(top5.indices[i]),
#           top5.values[i].item())
# ... ... ... ... 
# 1 ��s 18.585952758789062
# 2 ���� 16.70908546447754
# 3 �� 16.689165115356445
# 4 ���� 16.549083709716797
# 5 �u 16.422060012817383
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# >>> input = tokenizer("���{�̎�s�͂ǂ��ł����H", 
#                       return_tensors="pt")
# >>> tokens = model.generate(**input,
#                       max_new_tokens=10,
#                       do_sample=False)
# ... ... Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
# >>> tokenizer.decode(tokens[0], skip_special_tokens=True)
# '���{�̎�s�͂ǂ��ł���?\n�u�����v�Ƃ����s�s���A�Ȃ��u'
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# >>> input = tokenizer("�����͓V�C���悢�ł���\n" + 
#                       "�����ł���\n" +
#                       "�ǂ����֍s���܂��傤���B", 
#                       return_tensors="pt")
# >>> tokens = model.generate(**input,
#                       max_new_tokens=20,
#                       do_sample=False)
# ... ... Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
# >>> 
# >>> tokenizer.decode(tokens[0], skip_special_tokens=True)
# '�����͓V�C���悢�ł���\n�����ł���\n�ǂ����֍s���܂��傤���B\n���āA\n�����́A\n�u  �����V�C �v\n�ł��B\n�����́A\n'
# --------------------------------------------------------------------

