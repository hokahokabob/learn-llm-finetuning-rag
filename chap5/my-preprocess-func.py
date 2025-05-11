# -*- coding: sjis -*-

from janome.tokenizer import Tokenizer

t = Tokenizer()

def my_preprocess_func(text):
    keywords = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if (pos in ["名詞", "動詞", "形容詞"]):
            keywords.append(token.surface)
    return keywords

text = "私は小学生の頃大きな犬を飼っていました。"

print(my_preprocess_func(text))
# --> ['私', '小学生', '頃', '犬', '飼っ', 'い']
