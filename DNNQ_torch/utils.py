import torch 
import torch.nn.functional as F
import kaldi_io
import kaldiio
import os
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable

def get_data_label(label_path,data_path):
    label_data = {}
    feature_39d = {}
    data_list = []
    label_list = []
    count = 0
    #############################################################################################
    files_label = os.listdir(label_path)
    #read the label from the directory
    for item in files_label:
        f = kaldiio.load_ark(label_path + '/' + item )
        for key, value in f:
            label_data[key] = value
    #read the featrues and load into hash, make it faster for searching
    for key, value in data_path:
        feature_39d[key] = value
    #############################################################################################
    #make two list to save the features and label, remove the key.
    for data_key, data_value in feature_39d.items():
        data_label = label_data[data_key]
        if np.shape(data_value)[0] == np.shape(data_label)[0]:#save only the data, which is clean.
            for i in range(np.shape(data_value)[0]):
                #print(np.shape(data_value)[0])
                data_vector = data_value[i]
                data_list.append(data_vector)
                label_vector = data_label[i]
                label_list.append(int(label_vector))
        else:
            count+=1#count the dirty key
    return label_list,data_list
#not used
def to_categorical(y,num_classes):
    return np.eye(num_classes,dtype='uint8')
#dataloader, copy from the template             
class compset(Dataset):
    def __init__(self,data_list,label_list):
        self.data_list = data_list
        self.label_list = label_list
        
        
    def __getitem__(self, index):
        '''
        input :index 
        output :gt feature
        '''
        feature = self.data_list[index]
        label  = self.label_list[index]
        data = (feature,label)       
        return data


    def __len__(self):
        return len(self.data_list)


class dnnq(nn.Module):

    def __init__(self):
        super(dnnq,self).__init__()
        num_neurons = 512
        codebook_size = 127
        """
        4 full connected layer with ReLu activation function
        last layer is added a 0.25 dropout to aviod overfitting
        structure changed from Tobias Watzel in TUM MMK
        """
        self.dense1 = torch.nn.Linear(39, num_neurons)
        self.dense1_bn = torch.nn.BatchNorm1d(num_neurons)
        self.dense2 = torch.nn.Linear(num_neurons, num_neurons)
        self.dense2_bn = torch.nn.BatchNorm1d(num_neurons)
        self.dense3 = torch.nn.Linear(num_neurons, num_neurons)
        self.dense3_bn = torch.nn.BatchNorm1d(num_neurons)
        self.dense4 = torch.nn.Linear(num_neurons, num_neurons)
        self.dense4_bn = torch.nn.BatchNorm1d(num_neurons)
        self.dense5 = torch.nn.Linear(num_neurons,codebook_size)
        self.dense5_bn = torch.nn.BatchNorm1d(codebook_size)
        self.dropout = torch.nn.Dropout(0.25)


    def forward(self,x):
        
        x = F.relu(self.dense1(x.cuda().float()))
        x = self.dense1_bn(x)
        x = F.relu(self.dense2(x))
        x = self.dense2_bn(x)
        x = F.relu(self.dense3(x))
        x = self.dense3_bn(x)
        x = F.relu(self.dense4(x))
        x = self.dense4_bn(x)
        x = F.relu(self.dense5(x))
        x = self.dense5_bn(x)
        x = self.dropout(x)

        return x 