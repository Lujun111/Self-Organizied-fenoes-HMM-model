import os
import numpy as np
from collections import defaultdict
import re
##############################################################################################################
#this file is used to read the two singleton average and the most frequent file!!!!
path_averg = "/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/averg"
path_freq = "/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/freq"

files_averg = os.listdir(path_averg)
files_freq = os.listdir(path_freq)

save_aver_length = defaultdict(list)
save_freq_length = defaultdict(list)

for item in files_averg:
    f = np.load(path_averg +'/'+item)
    name_number = re.findall("\d+",item)
    save_aver_length[str(name_number[0])] = f

for item in files_freq:
    f = np.load(path_freq +'/'+item)
    name_number = re.findall(r"\d+",item)
    save_freq_length[str(name_number[0])] = f

print(save_aver_length)
print(save_freq_length)