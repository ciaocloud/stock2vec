import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm


class Stage1Dataset(Dataset):
    def __init__(self, df, dep_var, cat_vars=None, cont_vars=None):
        if cat_vars is None:
            cat_vars = []
        if cont_vars is None:
            cont_vars = []
        df_cat, df_cont = df[cat_vars], df[cont_vars]
        codes = [content.cat.codes.values for col_name, content in df_cat.items()]
        self.codes = np.stack(codes, axis=1).astype(np.int64) #if cats else None
        conts = [content.values for col_name, content in df_cont.items()]
        self.conts = np.stack(conts, axis=1).astype(np.float32) #if conts else None
        self.y = df[dep_var].values[:, np.newaxis]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.codes[idx], self.conts[idx], self.y[idx]]

class Stage1Model(nn.Module):

    def __init__(self, emb_sizes, n_cont, hidden_sizes=[1024, 512], emb_dropout=0., 
                       dropout_prob=0., use_bn=True, out_size=1):
        super().__init__()
        self.embeds = nn.ModuleList([self.embedding(nclass, ndim)
                                            for nclass, ndim in emb_sizes])
        self.emb_dropout = nn.Dropout(emb_dropout)
        total_emb_dim = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = total_emb_dim, n_cont
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layer_sizes = [self.n_emb+self.n_cont] + hidden_sizes + [out_size]
        if isinstance(dropout_prob, float):
            dropout_prob = [dropout_prob] * len(layer_sizes)
        else:   
            dropout_prob.insert(0, 0.0)
#        self.bns, self.drops, self.linears = [], [], []
        layers = []
        print(layer_sizes)
        for i in range(len(layer_sizes)-1):
            print(i, len(layer_sizes))
            n_in, n_out = layer_sizes[i], layer_sizes[i+1]
            if use_bn and i > 0: #i < len(layer_sizes)-1:
                bn = nn.BatchNorm1d(n_in)
                layers.append(bn)
#            self.bns.append(bn)
            dropout, p = None, dropout_prob[i]
            if p > 0:
                dropout = nn.Dropout(p)
                layers.append(dropout)
#            self.drops.append(dropout)

            fc = nn.Linear(n_in, n_out)
            nn.init.kaiming_normal_(fc.weight.data)
            layers.append(fc)
#            self.linear.append(fc)
            if i < len(layer_sizes)-2:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def embedding(self, nc, nd, mu=0., sigma=0.01):
#       # Embedding with trunc_normal initialization
        emb = nn.Embedding(nc, nd)
        with torch.no_grad():
            emb.weight.normal_().fmod_(2).mul_(sigma).add_(mu)
        return emb

    def forward(self, x_cat, x_cont):
        if self.n_emb > 0:
            x = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_dropout(x)
        if self.n_cont > 0:
            xcont = self.bn_cont(x_cont)
            if self.n_emb > 0:
                x = torch.cat([x, xcont], 1)
            else:
                x = xcont
#        for bn, dropout, fc in zip(self.bns, self.drops, self.linears):
#            if bn:
#                x = bn(x)
#            x = dropout(x)
#            x = fc(x)
#            x = F.relu(x)
        x = self.layers(x)
        return x

def train_model(model, train_dl, val_dl=None, n_epochs=1, criterion=nn.MSELoss(),
                    lr=1e-2, weight_decay=0., one_cycle=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if one_cycle:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                steps_per_epoch=len(train_dl), epochs=n_epochs)
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for xb_cat, xb_cont, yb in tqdm(train_dl):
            optimizer.zero_grad()
            xb_cat, xb_cont, yb = xb_cat.to(device), xb_cont.to(device), yb.to(device)
            preds = model(xb_cat, xb_cont)
            loss = criterion(preds, yb.float())

            loss.backward()
            optimizer.step()
            if one_cycle:
                scheduler.step()
            losses.append(loss.item())
        avg_loss = np.mean(np.array(losses))
        print("Epoch {0:d}: training loss={1:.3f}".format(epoch+1, avg_loss))

        if val_dl is not None:
            model.eval()
            with torch.no_grad():
                loss_val = 0.
                for xb_cat, xb_cont, yb in tqdm(val_dl):
                    xb_cat, xb_cont, yb = xb_cat.to(device), xb_cont.to(device), yb.to(device)
                    preds = model(xb_cat, xb_cont)
                    loss_val += criterion(preds, yb)
                avg_loss = loss_val / len(val_dl)
                print("Epoch {0:d}: training loss={1:.3f}".format(epoch+1, avg_loss))

def predict(model, dataset, batch_size=None):
    if batch_size is None:
        batch_size = len(dataset)
    dl = DataLoader(dataset, batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    preds, targets = [], []
    for xb_cat, xb_cont, yb in tqdm(dl):
        with torch.no_grad():
            xb_cat, xb_cont = xb_cat.to(device), xb_cont.to(device)
            yb_hat = model(xb_cat, xb_cont)
            preds.append(yb_hat.cpu().data.numpy())
            targets.append(yb.cpu().data.numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    return preds, targets

