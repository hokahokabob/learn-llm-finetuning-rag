# -*- coding: sjis -*-

import datasets
dolly = datasets.load_dataset("databricks/databricks-dolly-15k")

print(dolly['train'])

# --> 
# Dataset({
#     features: ['instruction', 'context', 'response', 'category'],
#     num_rows: 15011
# })

dolly_ja = datasets.load_dataset("kunishou/databricks-dolly-15k-ja")

print(dolly_ja['train'])

# --> 
# Dataset({
#     features: ['index', 'instruction', 'input', 'output', 'category'],
#     num_rows: 15015
# })

