import numpy as np

def test(model, test_loader, criterion):
    model.eval()
    test_loss = []
    correct = 0
    data_size = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda().long().squeeze()
        output = model(data)
        test_loss.append(criterion(output, target).item()) # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        # print(output)
        # print(target)
        correct += pred.eq(target.view_as(pred)).sum().item()
        data_size+= data.size(0)
    loss = np.mean(test_loss)
    acc = correct / data_size
    print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.1f}%)\n'.format(loss, correct, 
        data_size, acc*100))

    return loss, acc