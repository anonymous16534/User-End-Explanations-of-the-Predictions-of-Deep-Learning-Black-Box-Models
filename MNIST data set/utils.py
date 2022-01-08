'''Some helper functions
'''
import torchvision
from torchvision import datasets, transforms
import random
from random import shuffle
from emnist import list_datasets
from emnist import extract_training_samples
from dataset import *
import numpy as np
import copy
import torch

random.seed(7)

normalize_with_imagenet_vals = {
    'mean':[0.485, 0.456, 0.406],
    'std':[0.229, 0.224, 0.225]}

# Get the original MNIST dataset
def get_mnist_dataset():
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
        trainset = datasets.MNIST('./data', train=True, download=False,
                        transform=transform)
        testset = datasets.MNIST('./data', train=False, download=False,
                        transform=transform)
        return trainset, testset

def get_emnist_dataset():
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
        images, labels = extract_training_samples('letters')
        n = len(labels)
        sampled_indices = np.random.choice(n, 300, replace=False)
        images = images[sampled_indices]
        labels = labels[sampled_indices]
        trainset = Dataset(inputs = images, labels = labels, 
                transform= transform)

        return trainset

def get_attacker_dataset(dataset, labels, watermarked = False, watermark_ratio = 0.0):
        watermarkset = None
        watermark_inputs = []
        watermark_labels = []
        number_of_classes = max(labels)
        if watermarked:
                n = len(labels)
                m = int(watermark_ratio * n)
                watermark_indices = np.random.choice(n, m, replace=False)
                for idx in watermark_indices:
                        l = copy.deepcopy(labels[idx])
                        while labels[idx] == l:
                                r = torch.randint(0, 
                                10, (1,))
                                labels[idx] = r 
                        watermark_labels.append(labels[idx])
                        watermark_inputs.append(dataset[idx])

                watermarkset = AttackerDataset(dataset=watermark_inputs,
                                labels = watermark_labels)
        
        dataset = AttackerDataset(dataset = dataset,
        labels = labels)
        return dataset, watermarkset

def sample_imagenet(number_of_samples = None):

        transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(normalize_with_imagenet_vals['mean'], 
                normalize_with_imagenet_vals['std'])])

        trainset = datasets.ImageNet( root='./data', 
                split='train', transform=transform_train, download=True)

def query_labels(model, data_loader):
        model.eval()
        labels = []
        for i, (data, target) in enumerate(data_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                labels.extend(pred.detach().cpu())
        return labels

def query_predictions(model, data_loader):
        model.eval()
        predictions = []
        for i, (data, target) in enumerate(data_loader):
                data, target = data.cuda(), target.cuda()
                output = model(data)
                predictions+= output.detach().cpu()
        return predictions

# Get average weights
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] 
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
   

    
          
