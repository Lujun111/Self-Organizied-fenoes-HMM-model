import re
import os
import sys
import numpy as np
import kaldiio
from collections import defaultdict
file_ = kaldiio.load_mat
d = kaldiio.load_ark("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/test1111.ark")
# m = kaldiio.load_ark("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/alitest.1")
# with kaldiio.open_like_kaldi('gunzip -c /usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/alitest.1.gz |','rb') as f:
#    g = kaldiio.load_ark(f)
#    for key1, vector1 in g:
#        print(key1)
#        print(vector1[1])
#       print(len(vector1))
def findkey(target_key,start,end):
    m = kaldiio.load_ark("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/raw_mfcc_train.39d1.ark")
    for key, value in m:
        if key == target_key:
            return value[start:end]
output = defaultdict(list)

for k,(key, vector) in enumerate(d):
    i = 0
    j = 0
    while i<len(vector) and j<len(vector):
        start = vector[i]
        while j<len(vector) and vector[j] == start:
            j = j + 1
        output[str(start)].append(findkey(key,i,j))
        #print(vector[i:j])
        i=j
    if k%100==0:
        print(k)
        print(output)


print(type(output))

#    print(len(vector))
#    for i in range(len(vector)):
#        print((vector[i]))
#        print(i)
#        print(len(vector))
# for key, vector in m:
#     print(key)
#     print(vector)
#     print(len(vector))
    
#f = open("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/ali_1.txt")
#f1 = open("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/39d1.txt")
#line = f.readlines()
#line1 = f1.readlines()
#print(line[0][36])
#print(line[1][34:37])

    
#f1.close()
#f.close()