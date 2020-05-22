# ref: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
    
class TemporalDataset(Dataset):
    def __init__(self, X, y, timesteps=None):
        if timesteps is None:
            self.X = X
        else:
            self.X = X[:, -timesteps:]
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out+res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1)*dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out[:, :, -1]

def train_model(model, train_dl, val_dl=None, n_epochs=1, criterion=nn.MSELoss(),
                    lr=1e-2, weight_decay=0., one_cycle=False, device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(device)
    if one_cycle:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                steps_per_epoch=len(train_dl), epochs=n_epochs)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(train_dl):
            optimizer.zero_grad()
#             print(xb.size(), yb.size())
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb.float())

            loss.backward()
            optimizer.step()
            if one_cycle:
                scheduler.step()
            losses.append(loss.item())
        avg_loss = np.mean(np.array(losses))
        print("Epoch {0:d}: training loss={1:.5f}".format(epoch+1, avg_loss))

        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                loss_val = 0.
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    loss_val += criterion(preds, yb)
                avg_loss = loss_val / len(val_dl)
                print("Epoch {0:d}: val loss={1:.5f}".format(epoch+1, avg_loss))

def predict(model, dataset, batch_size=None):
    if batch_size is None:
        batch_size = len(dataset)
    dl = DataLoader(dataset, batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    preds, targets = [], []
    for xb, yb in tqdm(dl):
        with torch.no_grad():
            yb_hat = model(xb.to(device))
#             print(yb_hat.shape)
            preds.append(yb_hat.cpu().data.numpy())
            targets.append(yb.cpu().data.numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    return preds, targets
