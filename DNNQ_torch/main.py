import os
import torch.nn as nn
import torchvision
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from utils import get_data_label,dnnq,compset,to_categorical
from torch.autograd import Variable
import torch.nn.functional as F
import kaldiio
import numpy as np
######################################################################################
#setting for model,optimizer,cost and epochs
device = torch.device(a.is_av"cuda" if torch.cudailable() else "cpu")
model = dnnq().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
cost = nn.CrossEntropyLoss()
n_epochs = 15
#######################################################################################
#forget this one
# if device.type=='cuda':
#     model.load_state_dict(torch.load("./model_parameter.pkl"))
# else:
#     model.load_state_dict(torch.load("./model_parameter.pkl",map_location='cpu'))
#######################################################################################
#load train and validation data from ark file
label_path = "/usr/home/zhou/DNNQ_torch/dnnq_labels_neukirchen"
data_path = kaldiio.load_ark("/usr/home/zhou/kaldi/egs/tedlium/s5_r2/data/train/backup/raw_mfcc_train.39d1.ark")

val_label_path = "/usr/home/zhou/DNNQ_torch/dev"
val_data_path = kaldiio.load_ark("/usr/home/zhou/DNNQ_torch/val_39d.ark")
#import pdb;pdb.set_trace()
label_list,data_list = get_data_label(label_path,data_path)
val_label_list,val_data_list = get_data_label(val_label_path,val_data_path)
#################################################################################################
#this is for debug
# a_1 = np.random.rand(1,39)
# a_2 = np.random.rand(1,39)
# a_3 = np.random.rand(1,39)

# data_list=[a_1,a_2,a_3]
# label_list = [0,1,2]
# #label_list = to_categorical(label_list_1,3)

# val_data_list=[a_1,a_2,a_3]
# val_label_list = [0,1,2]
#val_label_list = to_categorical(val_label_list_1,3)
################################################################################################
#Dataloader for network, compset is a class for dataload
train = compset(data_list,label_list)
val = compset(val_data_list,val_label_list)
#set batch size and schuffle
data_train = torch.utils.data.DataLoader(dataset=train, batch_size = 131072, shuffle = True)
data_val = torch.utils.data.DataLoader(dataset=val, batch_size = 1, shuffle = True)
print('finish dataloading')
#every 5 round go into validatian
for epoch in range(1, n_epochs+1):
    if epoch%5 ==0:
        phase = 'val'
        model.eval()
    else:
        model.train()
        phase = 'train'
    
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch, n_epochs))
    print("-"*10)
#training step,you can set breakpoint to see, X_train is vector, y_train is label
    if(phase == 'train'):
        for data in data_train:
            X_train, y_train = data[0].cuda(),data[1].cuda()
            outputs= model(X_train).squeeze(1)
            loss = cost(outputs, y_train)
            print("-----loss is {}".format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data
#validation, calculation the loss
    elif(phase == 'val'):
        for data in data_val:
            X_val, y_val = data[0].cuda(), data[1].cuda()
            outputs = model(X_val).squeeze(1)
            loss = cost(outputs, y_val)

    print("Loss for {} is:{:.8f}".format(phase,running_loss/len(data_train)))
#save the model parameter.   
torch.save(model.state_dict(),'/usr/home/zhou/DNNQ_torch/model_last.pkl')
