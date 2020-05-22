import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

class HybridDataset(Dataset):
    def __init__(self, Xcat, Xcont, Xseq, y, timesteps=None):
        if timesteps is None:
            self.Xseq = Xseq[:, :, np.newaxis]
        else:
            self.Xseq = Xseq[:, -timesteps:, np.newaxis]
        self.Xseq = torch.from_numpy(self.Xseq.astype(np.float)).type(torch.Tensor)
        self.Xcat = torch.from_numpy(Xcat.astype(np.int))
        self.Xcont = torch.from_numpy(Xcont.astype(np.float)).type(torch.Tensor)
        self.y = torch.from_numpy(y.astype(np.float)).type(torch.Tensor)

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return [self.Xcat[idx], self.Xcont[idx], self.Xseq[idx], self.y[idx]]

class LSTMHybrid(nn.Module):
    def __init__(self, emb_sizes, n_cont, hidden_sizes=[1024, 512],
                 emb_dropout=0., dropout_prob=0., use_bn=True, out_size=1,
                 rnn_input_size=1, rnn_hidden_size=30, 
                 rnn_num_layers=2, rnn_output_size=None, rnn_dropout=0.):
        super().__init__()
        ## RNN Module
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        
        self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                           batch_first=True, num_layers=rnn_num_layers,
                           dropout=rnn_dropout)
        self.fc_rnn = nn.Linear(rnn_hidden_size, rnn_output_size)      

        ## Categorical Embedding Module
        self.embeds = nn.ModuleList([self.embedding(nclass, ndim)
                                        for nclass, ndim in emb_sizes])
        self.emb_dropout = nn.Dropout(emb_dropout)
        n_emb = sum(e.embedding_dim for e in self.embeds)
#         self.n_emb, self.n_cont = n_emb, n_cont

        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        ## Concatenate Continuous Features with Categorical & Temporal Feature Map
#        n_temporal = max(1, min(rnn_output_size, n_emb+n_cont)//2)
        layer_sizes = [n_emb+n_cont+rnn_output_size]+hidden_sizes+[out_size]
        if isinstance(dropout_prob, float):
            dropout_prob = [dropout_prob] * len(layer_sizes)
        else:   
            dropout_prob.insert(0, 0.0)
        layers = []
        for i in range(len(layer_sizes)-1):
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            if use_bn and i > 0:
                bn = nn.BatchNorm1d(n_in)
                layers.append(bn)
            dropout, p = None, dropout_prob[i]
            if p > 0:
                dropout = nn.Dropout(p)
                layers.append(dropout)
            fc = nn.Linear(n_in, n_out)
            nn.init.kaiming_normal_(fc.weight.data)
            layers.append(fc)
            if i < len(layer_sizes)-2:
                layers.append(nn.ReLU(inplace=True))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont, x_seq, hidden):
        x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = self.emb_dropout(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)

        h0, c0 = hidden
        out, (h, c) = self.rnn(x_seq, (h0.detach(), c0.detach()))
        out = self.fc_rnn(out[:, -1, :])

        x = torch.cat([x, out], 1)
        x = self.fc_layers(x)
        return x

    def embedding(self, nc, nd, mu=0., sigma=0.01):
        emb = nn.Embedding(nc, nd)
        with torch.no_grad():
            emb.weight.normal_().fmod_(2).mul_(sigma).add_(mu)
        return emb

    def init_hidden(self, batch_size, device):
        h0 = torch.randn(self.rnn_num_layers, batch_size,
                         self.rnn_hidden_size).requires_grad_().to(device)
        c0 = torch.randn(self.rnn_num_layers, batch_size,
                         self.rnn_hidden_size).requires_grad_().to(device)
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
        for xb_cat, xb_cont, xb_seq, yb in tqdm(train_dl):
#         for xb_cat, xb_cont, yb in train_dl:
            optimizer.zero_grad()
#             print(xb.size(), yb.size())
            xb_cat, xb_cont, xb_seq, yb = xb_cat.to(device), xb_cont.to(device), xb_seq.to(device), yb.to(device)
            hidden = model.init_hidden(xb_seq.size(0), device)
            preds = model(xb_cat, xb_cont, xb_seq, hidden)
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
                for xb_cat, xb_cont, xb_seq, yb in val_dl:
                    xb_cat, xb_cont, xb_seq, yb = xb_cat.to(device), xb_cont.to(device), xb_seq.to(device), yb.to(device)
                    hidden = model.init_hidden(xb_seq.size(0), device)
                    preds = model(xb_cat, xb_cont, xb_seq, hidden)
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
    for xb_cat, xb_cont, xb_seq, yb in tqdm(dl):
        with torch.no_grad():
            xb_cat, xb_cont, xb_seq = xb_cat.to(device), xb_cont.to(device), xb_seq.to(device)
            hidden = model.init_hidden(xb_seq.size(0), device)
            yb_hat = model(xb_cat, xb_cont, xb_seq, hidden)            
            preds.append(yb_hat.cpu().data.numpy())
            targets.append(yb.cpu().data.numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    return preds, targets

