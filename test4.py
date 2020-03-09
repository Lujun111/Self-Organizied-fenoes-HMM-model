import re
import os
import sys
import numpy as np
import kaldiio
from collections import defaultdict
# from  KaldiHelper.MiscHelper import *
import kaldi_io
from KaldiHelper import MiscHelper
import pandas as pd
import json
######################################################################################################################
#inital
output = defaultdict(list)
output_2 = defaultdict(list)

average_frame = defaultdict(int)
frequent_frame = defaultdict(int)
save_aver_length = defaultdict(list)
save_freq_length = defaultdict(list)

feature_39d = {}
phone_39d = {}
######################################################################################################################
######################################################################################################################
#read the two ark file with kaldiio
phone_166 = kaldiio.load_ark("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/test1111.ark")
m = kaldiio.load_ark("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/raw_mfcc_train.39d1.ark")
#save it as a HASH, time complexity down to O(1)
for key, value in m:
    feature_39d[key] = value

for key,value in phone_166:
    phone_39d[key] = value

######################################################################################################################
# find_dimension function is used to find the diemension of the wanted vector    
def find_dimension(target_key,start,end,feature_39d):
    value = feature_39d[target_key]
    return value[start:end].shape[0]
# find_key function is used to return the wanted vector
def find_key(target_key,start,end,feature_39d):
    value = feature_39d[target_key]
    return value[start:end]


########################################################################################################################                                                                                                                  
########################################################################################################################

#use quick and slow point here to find the the dic for the shape and dic for the vector
for k,(key, vector) in enumerate(phone_39d.items()):
    M = MiscHelper.Misc(1,2,3,4,5)#first use the tobias' mapping to map 166 phone_id to 40
    M._get_transformation_vec()
    i = 0
    j = 0
    while i<len(vector) and j<len(vector):
        vector1 = M.trans_vec_to_phones(vector)
        start = vector1[i]
        while j<len(vector1) and vector1[j] == start:
            j = j + 1
        output[str(start)].append(find_dimension(key,i,j,feature_39d))#dic for the shape
        output_2[str(start)].append(find_key(key,i,j,feature_39d))#dic for the vector
        i=j
########################################################################################################################                                                                                                                  
########################################################################################################################    
#save two new dics for save the average length of the frame and the most frequent length of the frame
for kk, value in output.items():
    average_frame[str(kk)] = int(np.mean(value))            
    frequent_frame[str(kk)] = pd.Series(data=value).mode()[0]

########################################################################################################################                                                                                                                  
########################################################################################################################
#find the vector in dic output2 according to the average length 
for key3, vector3 in output_2.items():
    average_length = average_frame[key3]
    for i in range(len(vector3)):
        if vector3[i].shape[0] == average_length:
            save_aver_length[str(key3)] = vector3[i]
            break

########################################################################################################################                                                                                                                  
########################################################################################################################
#find the vector in dic output2 according to the most frequent length 
for key4, vector4 in output_2.items():
    frequent_length = frequent_frame[key4]
    for i in range(len(vector4)):
        if vector4[i].shape[0] == frequent_length:
            save_freq_length[str(key4)] = vector4[i]
            break
########################################################################################################################                                                                                                                  
########################################################################################################################
#save the two vectors dics it in the folder
# for key ,value in save_freq_length.items():
#     file_name  = './freq/{}.npy'.format(key)
#     np.save(file_name,value)

# for key ,value in save_aver_length.items():
#     file_name  = './averg/{}.npy'.format(key)
#     np.save(file_name,value)

########################################################################################################################                                                                                                                  
########################################################################################################################
#save the two length as txt in the folder 
js1 = json.dumps(average_frame)
js2 = json.dumps(frequent_frame)
f1 = open('average_frame.txt','w')
f2 = open('frequent_frame.txt','w')
f1.write(js1)
f2.write(js2)

