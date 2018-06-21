import glob
import numpy as np
import torch
import torch.nn as nn
from settings import BASE_PATH
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


weight_shapes = [(11, 1, 5, 5),
                 (24, 11, 5, 5),
                 (274, 384),
                 (10, 274)]

bias_shapes = [(11, 1),
               (24, 1),
               (274, 1),
               (10, 1)]

weight_files = glob.glob('vals/lr*ep18*wt.txt')
weight_files.sort()

bias_files = glob.glob('vals/lr*ep18*bs.txt')
bias_files.sort()

weights = [torch.from_numpy(np.loadtxt(fname).reshape(shape)).float() for fname,
           shape in zip(weight_files, weight_shapes)]

biases = [torch.from_numpy(np.loadtxt(fname).reshape(shape)).float() for fname,
          shape in zip(bias_files, bias_shapes)]


def proc(x):
    x = F.conv2d(x, weights[0], biases[0].view(-1))
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    x = F.conv2d(x, weights[1], biases[1].view(-1))
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    x = x.view(x.size(0), -1)

    x = F.linear(x.view(x.size(0), -1), weights[2], biases[2].view(-1))
    x = F.relu(x)

    x = F.linear(x, weights[3], biases[3].view(-1))

    return x


kwargs = {'num_workers': 1, 'pin_memory': True}
dataset_path = BASE_PATH + 'mnist' + '_data'
ds = datasets.MNIST
test_loader = torch.utils.data.DataLoader(
    ds(dataset_path, train=False, transform=transforms.Compose([
        transforms.ToTensor(), lambda x: 2 * (x - 0.5),
    ])),
    batch_size=128, shuffle=True, **kwargs)

discrimination_loss = nn.functional.cross_entropy
test_loss = 0
correct = 0

for data, target in test_loader:
    data, target = Variable(data), Variable(target)
    output = proc(data)
    test_loss += discrimination_loss(output,
                                     target, size_average=False).item()
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
test_loss /= len(test_loader.dataset)
print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100.0 * float(correct) / len(test_loader.dataset)))
