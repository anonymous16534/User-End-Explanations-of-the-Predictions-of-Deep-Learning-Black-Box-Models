import os
import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, inputs, labels, transform = None):
      self.inputs = inputs
      self.labels = labels
      self.transform = transform         
    def __getitem__(self, index):
        img = self.inputs[index]
        label = self.labels[index]
        img = self.transform(img)
        return img, label 
    def __len__(self):
        return len(self.labels)

class AttackerDataset(data.Dataset):
    def __init__(self, dataset, labels, transform = None):
      self.dataset = dataset
      self.labels = labels
      self.transform = transform   
    def __getitem__(self, index):
        img = self.dataset[index][0]
        label = self.labels[index]
        return img, label 
    def __len__(self):
        return len(self.labels)
    

# A method for combining datasets  
def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)
    