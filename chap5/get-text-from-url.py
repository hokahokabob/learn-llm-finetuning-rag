# -*- coding: sjis -*-

import requests
from bs4 import BeautifulSoup

url = 'https://www.aozora.gr.jp/cards/000035/files/275_13903.html'

# URL����y�[�W�̓��e���擾
response = requests.get(url)

# HTML�̉��
soup = BeautifulSoup(response.content, 'html.parser')

# �s�v�ȃ^�O������
for script_or_style in soup(['script', 'style']):
    script_or_style.extract()

# ���ׂẴe�L�X�g�𒊏o
text = soup.get_text()

# �s�𕪊����āA�擪�Ɩ����̋󔒂��폜
lines = (line.strip() for line in text.splitlines())

# ��̍s������
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

# �e�L�X�g����̕�����Ɍ���
text = '\n'.join(chunk for chunk in chunks if chunk)

# �e�L�X�g���t�@�C���ɕۑ�
outtext = 'joseito.txt'
with open(outtext, 'w', encoding='utf-8') as file:
    file.write(text)

print(url, "�̓��e��",outtext,"�ɏo�͂��܂���")
