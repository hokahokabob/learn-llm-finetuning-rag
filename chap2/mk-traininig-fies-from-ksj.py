# -*- coding: sjis -*-

import zipfile
import codecs

ftrain = codecs.open('train.txt', 'wb', 'cp932', 'ignore')
fval = codecs.open('val.txt', 'wb', 'cp932', 'ignore')
ftest = codecs.open('test.txt', 'wb', 'cp932', 'ignore')

s, n, flg  = "", 0, 0
with zipfile.ZipFile('KSJ.zip') as zf:
    for file in zf.namelist():
        with zf.open(file) as f:
            for line in f:
                line = line.decode('utf-8')
                w, _  = line.strip().split()
                s += w
                if (w == "。"):
                    s += "\n"
                    n += 1
                    if (n % 10 == 0):
                        if (flg  == 0):
                            fval.write(s)
                            s = ""
                            flg = 1
                        else:
                            ftest.write(s)
                            s = ""
                            flg = 0
                    else:
                        ftrain.write(s)
                        s = ""

print("train.txt、test.txt、val.txt を作成しました")












        
