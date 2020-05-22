import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

class CrossDataset(Dataset):
    def __init__(self, df, ticker, df_pred, dep_var, cat_vars=None, cont_vars=None, mkts=None):
        df = df[df['ticker']==ticker]
        if cat_vars is None:
            cat_vars = []
        if cont_vars is None:
            cont_vars = []
#         cat_keep, cat_size = [], []
#         for c in cat_vars:
#             sz = 
#             if len(df[c].unique()) > 1:
#                 cat_keep.append(c)
#                 cat
#         cat_sz = [(c, len(df[c].cat.categories)+1
#         df_cat, df_cont = df[cat_keep], df[cont_vars]
        df_cat, df_cont = df[cat_vars], df[cont_vars]
        codes = [content.cat.codes.values for col_name, content in df_cat.items()]
        self.codes = np.stack(codes, axis=1).astype(np.int64)
        conts = [content.values for col_name, content in df_cont.items()]
        self.conts = np.stack(conts, axis=1).astype(np.float32)
        df_pred = df_pred.iloc[df_pred.index==df.Date]
        if mkts is None:
            df_pred = df_pred.dropna(axis=1)
        else:
            df_pred = df_pred[mkts]
        self.mkts = df_pred.columns.to_list()
        preds = [content.values for col_name, content in df_pred.items()]
        self.preds = np.stack(preds, axis=1).astype(np.float32)
        self.y = df[dep_var].values[:, np.newaxis]
        
    def get_mkts(self):
        return self.mkts
    
#     def get_dims(self):
#         return self.codes.shape[1], self.conts.shape[1], self.preds.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.codes[idx], self.conts[idx], self.preds[idx], self.y[idx]]

class CrossModel(nn.Module):
    def __init__(self, emb_sizes, n_cont, n_mkt, hidden_sizes=[128, 64], 
                    emb_dropout=0., dropout_prob=0., use_bn=True, out_size=1):
        super().__init__()
        self.embeds = nn.ModuleList([self.embedding(nclass, ndim) for nclass, ndim in emb_sizes])
        self.emb_dropout = nn.Dropout(emb_dropout)
        total_emb_dim = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont, self.n_mkt = total_emb_dim, n_cont, n_mkt
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        self.bn_mkt = nn.BatchNorm1d(n_mkt)
        self.mkt_fc = nn.Linear(n_mkt, 20)
        nn.init.kaiming_normal_(self.mkt_fc.weight.data)

        layer_sizes = [self.n_emb+self.n_cont+20] + hidden_sizes + [out_size]
        if isinstance(dropout_prob, float):
            dropout_prob = [dropout_prob] * len(layer_sizes)
        else:   
            dropout_prob.insert(0, 0.0)
#        self.bns, self.drops, self.linears = [], [], []
        layers = []
#         print(layer_sizes)
        for i in range(len(layer_sizes)-1):
#             print(i, len(layer_sizes))
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

    def forward(self, x_cat, x_cont, x_mkt):
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
        if self.n_mkt > 0:
            xmkt = self.bn_mkt(x_mkt)
            xmkt = F.relu(self.mkt_fc(xmkt))
            x = torch.cat([x, xmkt], 1)
        x = self.layers(x)
        return x
    
    
def train_model(model, train_dl, val_dl=None, n_epochs=1, criterion=nn.MSELoss(),
                    lr=1e-2, weight_decay=0., one_cycle=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=lr)#, weight_decay=weight_decay)
    if one_cycle:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                steps_per_epoch=len(train_dl), epochs=n_epochs)
    model.to(device)
    for epoch in range(n_epochs):
        model.train()
        losses = []
#         for xb_cat, xb_cont, xb_mkt, yb in tqdm(train_dl):
        for xb_cat, xb_cont, xb_mkt, yb in train_dl:
            optimizer.zero_grad()
            xb_cat, xb_cont, xb_mkt, yb = xb_cat.to(device), xb_cont.to(device), xb_mkt.to(device), yb.to(device)
            preds = model(xb_cat, xb_cont, xb_mkt)
            loss = criterion(preds, yb.float())
#             print(loss)
            l1reg = 0.
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1reg += torch.sum(abs(param))
#             print(l1reg)
            loss += weight_decay * l1reg

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
                for xb_cat, xb_cont, xb_mkt, yb in tqdm(val_dl):
                    xb_cat, xb_cont, xb_mkt, yb = xb_cat.to(device), xb_cont.to(device), xb_mkt.to(device), yb.to(device)
                    preds = model(xb_cat, xb_cont, xb_mkt)
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
    for xb_cat, xb_cont, xb_mkt, yb in tqdm(dl):
        with torch.no_grad():
            xb_cat, xb_cont, xb_mkt = xb_cat.to(device), xb_cont.to(device), xb_mkt.to(device)
            yb_hat = model(xb_cat, xb_cont, xb_mkt)
            preds.append(yb_hat.cpu().data.numpy())
            targets.append(yb.cpu().data.numpy())
    preds = np.vstack(preds)
    targets = np.vstack(targets)
    return preds, targets