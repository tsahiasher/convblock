from conv_block import ConvBlock
import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


def load_MNIST(train=True, transform=None):
    root = './datasets/'
    download = True
    MNIST_data = torchvision.datasets.MNIST(root=root, train=train, transform=transform, download=download)
    return MNIST_data


def create_MNIST_datasets(HP):
    transform = HP['transform']
    split_lengths = HP['split_lengths']

    dataset = load_MNIST(train=True, transform=transform)
    lengths = (np.array(split_lengths) * len(dataset)).astype(int)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)
    test_dataset = load_MNIST(train=False, transform=transform)

    return train_dataset, val_dataset, test_dataset


loss_fn = nn.CrossEntropyLoss()


def parameters_norm(net):
    pars = [list(p.view(-1).detach().numpy()) for p in net.parameters()]
    all = np.array([p for sublist in pars for p in sublist])
    norm = np.sqrt(np.sum(all ** 2))
    return norm


def evaluation(HP, net, val_data):
    net.eval()
    with torch.no_grad():
        losses = []
        for batch in val_data:
            y_hat = net(batch[0])
            y = batch[1]
            losses.append(loss_fn(y_hat, y).item())
    return np.mean(losses)


def train(HP, net, train_dataset, val_dataset, optimizer):
    train_losses = []
    val_losses = []
    norms = []
    val_data = DataLoader(val_dataset, batch_size=HP['batch_size'])

    for epoch in range(HP['epochs']):

        train_data = DataLoader(train_dataset, batch_size=HP['batch_size'], shuffle=True)

        net.train()

        batch_losses = []
        for batch_idx, batch in enumerate(train_data):
            #print(batch_idx)
            optimizer.zero_grad()
            y_hat = net(batch[0])
            train_loss = loss_fn(y_hat, batch[1])
            train_loss.backward()
            batch_losses.append(train_loss.item())
            optimizer.step()

        train_losses.append(np.mean(batch_losses))
        val_losses.append(evaluation(HP, net, val_data))
        norm = parameters_norm(net)
        norms.append(norm)
        print('epoch', epoch, 'train_loss', train_losses[-1], 'val_loss', val_losses[-1], 'norm', norm)

    return train_losses, val_losses, norms


HP = {}
HP['lr'] = 0.02
HP['epochs'] = 20
HP['batch_size'] = 64
HP['transform'] = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])
HP['split_lengths'] = [0.8, 0.2]

train_dataset, val_dataset, test_dataset = create_MNIST_datasets(HP)
out_channels = 8
cb = ConvBlock(in_channels=1, out_channels=out_channels, conv_block='ConvActivBnFc', pool_kernel=2, pool_stride=2,
               padding='SAME', causal=True, separable=False, bn_init='ones', in_features=28*28, out_features = 10, fc_init='xavier')

# class fc(nn.Module):
#     def __init__(self, out_channels):
#         super(fc, self).__init__()
#         self.fc1 = torch.nn.Linear(out_channels * 14 * 14, 10, bias=True)
#     def forward(self, input):
#         return self.fc1(input.view(input.size(0), -1))
#
# fc1 = fc(out_channels)
# net = nn.Sequential(cb)
# net.append(fc1)

optimizer = torch.optim.SGD(cb.parameters(), lr=HP['lr'])

train_losses, val_losses, norms = train(HP, cb, train_dataset, val_dataset, optimizer)

print(cb)
