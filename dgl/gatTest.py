import argparse

import dgl
import dgl.nn as dglnn

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from gat import GAT as myGAT

from ogb.nodeproppred import DglNodePropPredDataset


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            myGAT(
                in_size,
                hid_size,
                heads[0],
            )
        )
        self.gat_layers.append(
            myGAT(
                hid_size * heads[0],
                out_size,
                heads[1]
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )


data = DglNodePropPredDataset(name="ogbn-proteins")
split_idx = data.get_idx_split()
g, labels = data[0]
g.ndata["label"] = labels


n_train = int(g.num_nodes() * 0.6)
n_val = int(g.num_nodes() * 0.2)
train_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
val_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
test_mask = torch.zeros(g.num_nodes(), dtype=torch.bool)
train_mask[:n_train] = True
val_mask[n_train: n_train + n_val] = True
test_mask[n_train + n_val:] = True
g.ndata["train_mask"] = train_mask
g.ndata["val_mask"] = val_mask
g.ndata["test_mask"] = test_mask
masks = [train_mask, val_mask, test_mask]

n_rows = g.num_nodes()
n_hidden = in_size = 8

features = torch.tensor(np.random.rand(n_rows, in_size), dtype=torch.float)
labels = torch.tensor(np.random.randint(0, 8, size=n_rows), dtype=torch.long)

# create GAT model
# in_size = features.shape[1]
# out_size = data.num_classes
model = GAT(in_size, 8, 8, heads=[8, 1])


# model training
print("Training...")
train(g, features, labels, masks, model)

# test the model
print("Testing...")
acc = evaluate(g, features, labels, masks[2], model)
print("Test accuracy: ", acc)