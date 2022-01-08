import torch.optim as optim
import numpy as np
import copy
from tqdm import tqdm
from tqdm import tqdm_notebook

def train(model, train_loader, criterion, optimizer, local_epochs):
    model.train()
    epochs_loss = []
    for epochs in tqdm_notebook(range(local_epochs)):
        epoch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda().long().squeeze()
            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()            
            optimizer.step()
            epoch_loss.append(loss.item())
        # print('Train epoch: {} \tLoss: {:.6f}'.format((global_round+1), 
        # np.mean(epoch_loss)))

        epochs_loss.append(np.mean(epoch_loss))
    return model, np.mean(epochs_loss)