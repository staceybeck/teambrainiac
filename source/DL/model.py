
import scipy.io
import os
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
import boto3
import tempfile
import tqdm
import random
from path_config import mat_path
from botocore.exceptions import ClientError
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Pytroch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset

#import torchvision.transforms as transforms
from torch.nn import ReLU, CrossEntropyLoss, Conv3d, Module, Softmax, AdaptiveAvgPool3d
from torch.optim import Adam, SGD

#from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    
    #Conv1
    self.conv1 = nn.Conv3d(in_channels = 1, 
                           out_channels = 32, 
                           kernel_size = (1,1,1), 
                           stride = (1,1,1)
                           )
    self.bn1 = nn.BatchNorm3d(32)
    self.conv2 = nn.Conv3d(in_channels = 32, 
                           out_channels = 64, 
                           kernel_size = (7,7,7),
                           stride = (2,2,2)
                           )
    self.bn2 = nn.BatchNorm3d(64)
    self.conv3 = nn.Conv3d(in_channels = 64, 
                           out_channels = 64, 
                           kernel_size = (3,3,3),
                           stride = (2,2,2)
                           )
    self.bn3 = nn.BatchNorm3d(64)
    self.conv4 = nn.Conv3d(in_channels = 64, 
                           out_channels = 128, 
                           kernel_size = (3,3,3),
                           stride = (2,2,2)
                           )
    self.bn4 = nn.BatchNorm3d(128) 
    self.pool1 = nn.AdaptiveAvgPool3d((1,1,1)) #Global Average Pool, takes the average over last two dimensions to flatten 
  
                                                             
    # Fully connected layer
    self.fc1 = nn.Linear(128,64) # need to find out the size where AdaptiveAvgPool 
    self.fc2 = nn.Linear(64, 2) # left with 2 for the two classes                     

  def forward(self, xb):
    xb = self.bn1(F.relu(self.conv1((xb))))
    xb = self.bn2(F.relu(self.conv2((xb)))) # Takes a long time
    xb = self.bn3(F.relu(self.conv3((xb))))
    xb = self.bn4(F.relu(self.conv4((xb))))
    xb = self.pool1(xb)
    xb = xb.view(xb.shape[:2])
    xb = self.fc1(xb)
    xb = self.fc2(xb)
    return xb    
  
  
  
  
  
  
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
  
  
  
  
  
  
def run_cnn(model, epochs, learning_rate, loss_func, opt, dl):
  metrics_dict = {}
  # Run Model
  for epoch in range(1, 1+epochs):
    i = 1
    accuracy_list = []
    loss_list = []
    model.train()
    print('epoch', epoch)
    for xb, yb in dl:
      print('batch', i)
      i += 1

      xb = xb.float()
      pred = model(xb)
      loss_batch = loss_func(pred, yb)
      loss_list.append(loss_batch)
      accuracy_batch = accuracy(pred, yb)
      if int(accuracy_batch) == 1:
        print('Perfect Accuracy\nStopping early to avoid overfitting\n\n')
        break


      accuracy_list.append(accuracy_batch)

      print('Batch Loss', loss_batch)
      print('Batch Accuracy', accuracy_batch)

      loss_batch.backward()
      opt.step()
      opt.zero_grad()

    model.eval()

    metrics_dict['epoch_'+str(epoch)] = {'accuracy':accuracy_list, 'loss':loss_list}

    print('epoch', epoch, 'finished\n')
    try:
      past_epoch_accuracies = [sum(metrics_dict['epoch_'+str(epoch-2)]['accuracy']), sum(metrics_dict['epoch_'+str(epoch-1)]['accuracy'])]
      current_epoch_accuracy = sum(metrics_dict['epoch_'+str(epoch)]['accuracy'])
      if past_epoch_accuracies[0] > current_epoch_accuracy and past_epoch_accuracies[1] > current_epoch_accuracy:
        print('Early stop to avoid overfitting\nModel accuracies did not decrease for two epochs')
        return model, metrics_dict

    except:
      pass
  
  return model, metrics_dict


