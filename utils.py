import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from models import *
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import gc


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def create_classes_train(dataset, localmodel,device):
    classes = []
    counter = 0
    for data, target in dataset:
        counter += 1

            # Send the data and label to the device
        data, target = data.to(device), target.to(device)
    #     print('the original was: ',target)
    #     counter += 1
    #     print(counter)
    #     data = data.to(device)
        with torch.no_grad():
            output=localmodel(data)#.argmax().item()
            # print(output.item())
            # output = output.to(device=device, dtype=torch.int64)
            # output = output.to('cpu')
    #         print(output)
            classes.append(output)
    return classes



def create_classes_val(dataset, localmodel,device):
    classes = []
    counter = 0
    for data, target in dataset:
        counter += 1

            # Send the data and label to the device
        data, target = data.to(device), target.to(device)
    #     print('the original was: ',target)
    #     counter += 1
    #     print(counter)
    #     data = data.to(device)
        with torch.no_grad():
            output=localmodel(data).argmax().item()
            # print(output.item())
            # output = output.to(device=device, dtype=torch.int64)
            # output = output.to('cpu')
    #         print(output)
            classes.append(output)
    return classes


def get_new_images(dataset, local_model,path, device):
    
    classes = []
    images = []
    counter = 0
    new_labels = []
    eps_all = []
    perturbed_examples = []
    counter = 0
    # print(path)
    for data, target in dataset:
    	print(counter)
    	data, target = data.to(device), target.to(device)
    	eps = 0.0
    	output = local_model(data)
    	if target == 0:
    		our_target = 1
    		our_target =torch.tensor([our_target])
    	else:
    		our_target = 0
    		our_target =torch.tensor([our_target])
    	our_target = to_device(our_target, device)
    	counter+=1
    	while True:
    		perturbed_image = data.clone()
    		perturbed_image.requires_grad = True
    		output = local_model(perturbed_image)
    		loss = F.nll_loss(output, our_target)
    		local_model.zero_grad()
    		loss.backward()
    		img_grad = perturbed_image.grad.data
    		perturbed_image = perturbed_image - eps*img_grad
    		output = local_model(perturbed_image)
    		new_label = output.max(1, keepdim=True)[1]
    		if(new_label.item() == our_target.item() and eps>0.0):
    			perturbed_examples.append(perturbed_image.squeeze().data.cpu().numpy())
    			new_labels.append(new_label)
    			eps_all.append(eps)
    			torchvision.utils.save_image(perturbed_image, path+str(counter)+'.png')
    			print("Image {} has been modified with epsilon {}".format(counter-1, eps))
    			break
    		if(new_label.item() == our_target.item() and eps==0.0):
    			break
    		eps += 0.05
    		if eps > 0.99:
    			break


def create_new_targets(local_test_loader):
    new_targets= []
    for data, target in local_test_loader:
        if target == 1:
            new_targets.append(0)
        else:
            new_targets.append(1)
    return new_targets



def test(original_model, device, local_test_loader):

    # Accuracy counter
    correct = 0
    wrong_examples = []
    logits = []
    labels = []
    counter = 0

    # Loop over all examples in test set
    for data, target in local_test_loader:
        counter += 1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Forward pass the data through the model

        with torch.no_grad():
            output = original_model(data)

        # Check for success
    
#         final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred = output.argmax()
        if final_pred.item() == target.item():
            correct += 1
        else:

            wrong_examples.append(data)
            labels.append(target)
            logits.append(output)
        if len(labels) > 300:
            break
    
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(local_test_loader))

    # Return the accuracy and an adversarial example
    return final_acc, wrong_examples, labels, logits


def test_images_true_classified(original_model, device, local_test_loader):

    # Accuracy counter
    correct = 0
    wrong_examples = []
    logits = []
    labels = []
    counter = 0

    # Loop over all examples in test set
    for data, target in local_test_loader:
        counter += 1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Forward pass the data through the model

        with torch.no_grad():
            output = original_model(data)

        # Check for success
    
#         final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred = output.argmax()
        if final_pred.item() == target.item():
        	wrong_examples.append(data)
        	labels.append(target)
        	logits.append(output)
        	correct += 1
    final_acc = correct/float(len(local_test_loader))

    # Return the accuracy and an adversarial example
    return final_acc, wrong_examples, labels, logits


