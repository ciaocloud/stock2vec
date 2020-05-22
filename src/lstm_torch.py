import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

# class TRNDataset(Dataset):
#     def __init__(self, X, y, timesteps=None):
#         if timesteps is None:
#             self.X = X[:, :, np.newaxis]
#         else:
#             self.X = X[:, -timesteps:, np.newaxis]
#         self.y = y.astype(np.float64)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return [self.X[idx], self.y[idx]]

class LSTM(nn.Module):
    def __init__(self, rnn_input_size=1, rnn_hidden_size=30, 
                 rnn_num_layers=2, rnn_output_size=1, rnn_dropout=0.):

        super().__init__()
        ## RNN Module
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        
#         self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_size,
#                            num_layers=rnn_num_layers, dropout=rnn_dropout)
        
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                           batch_first=True, num_layers=rnn_num_layers,
                           dropout=rnn_dropout)
        self.fc = nn.Linear(rnn_hidden_size, rnn_output_size)      

    def forward(self, x, hidden):
#         print("Input size:  ", x.size()) ## (batch_size, seq_len, num_seq)
        h0, c0 = hidden
        out, (h, c) = self.rnn(x, (h0.detach(), c0.detach()))
#         out, (h, c) = self.rnn(x, hidden) 
#         print("Output size:  ", out.size()) ##(batch_size, seq_len, hidden_size)
#         print(out.shape, h.shape)
#         print(out[0, -1, :], h[1, -1, :])
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, batch_size, device):
#         weight = next(self.parameters()).data
#         h0 = weight.new(self.rnn_num_layers, batch_size, self.rnn_hidden_size).zero_().to(device)
#         c0 = weight.new(self.rnn_num_layers, batch_size, self.rnn_hidden_size).zero_().to(device)
        h0 = torch.randn(self.rnn_num_layers, batch_size,
                         self.rnn_hidden_size).requires_grad_().to(device) # hidden state
        c0 = torch.randn(self.rnn_num_layers, batch_size, 
                         self.rnn_hidden_size).requires_grad_().to(device) # cell state 
        return (h0, c0)
        
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
#         for xb_cat, xb_cont, yb in train_dl:
            optimizer.zero_grad()
#             print(xb.size(), yb.size())
            hidden = model.init_hidden(xb.size(0), device)#.to(device)
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb, hidden)
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
                    hidden = model.init_hidden(xb.size(0), device)#.to(device)
                    preds = model(xb, hidden)
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
            hidden = model.init_hidden(xb.size(0), device)
            yb_hat = model(xb.to(device), hidden)
#             print(yb_hat.shape)
            preds.append(yb_hat.cpu().data.numpy())
            targets.append(yb.cpu().data.numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    return preds, targets
