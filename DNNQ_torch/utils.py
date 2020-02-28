import torch 
import torch.nn.functional as f
import kaldi_io
import kaldiio
import os
from collections import defaultdict

label_data = {}

label_path = "/usr/home/zhou/DNNQ_torch/dnnq_labels_neukirchen"
files_label = os.listdir(label_path)
for item in files_label:
    f = kaldiio.load_ark(label_path + '/' + item )
    for key, value in f:
        label_data[key] = value

print(label_data)















# class dnnq(nn.Module):

#     def __init__(self):
#         super(dnnq,self).__init__()
#         """
#         4 full connected layer with ReLu activation function
#         last layer is added a 0.25 dropout to aviod overfitting
#         structure changed from Tobias Watzel in TUM MMK
#         """
#         self.dense1 = torch.nn.Linear(self.feature, num_neurons)
#         self.dense1_bn = torch.nn.BatchNorm1d(num_neurons)
#         self.dense2 = torch.nn.Linear(num_neurons, num_neurons)
#         self.dense2_bn = torch.nn.BatchNorm1d(num_neurons)
#         self.dense3 = torch.nn.Linear(num_neurons, num_neurons)
#         self.dense3_bn = torch.nn.BatchNorm1d(num_neurons)
#         self.dense4 = torch.nn.Linear(num_neurons, num_neurons)
#         self.dense4_bn = torch.nn.BatchNorm1d(num_neurons)
#         self.dropout = torch.nn.Dropout(0.25)


#     def forward(self,x):
#         x = F.relu(self.dense1(x))
#         x = self.dense1_bn(x)
#         x = F.relu(self.dense2(x))
#         x = self.dense2_bn(x)
#         x = F.relu(self.dense3(x))
#         x = self.dense3_bn(x)
#         x = F.relu(self.dense4(x))
#         x = self.dense4_bn(x)
#         x = self.dropout(x)
#         return x 