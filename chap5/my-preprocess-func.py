# -*- coding: sjis -*-

from janome.tokenizer import Tokenizer

t = Tokenizer()

def my_preprocess_func(text):
    keywords = []
    for token in t.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        if (pos in ["����", "����", "�`�e��"]):
            keywords.append(token.surface)
    return keywords

text = "���͏��w���̍��傫�Ȍ��������Ă��܂����B"

print(my_preprocess_func(text))
# --> ['��', '���w��', '��', '��', '����', '��']
