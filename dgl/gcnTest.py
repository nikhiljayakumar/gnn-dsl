import time
import os
import numpy

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn import MyGraphConv
from dgl.nn import GraphConv as RealGraphConv

torch.set_printoptions(threshold=10000)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = MyGraphConv(in_feats, h_feats, norm="both")
        self.conv2 = MyGraphConv(h_feats, num_classes, norm="both")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0
    
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    for epoch in range(100):
        # Forward
        logits = model.forward(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
        )
    
start = time.time()
# provided cora dataset
dataset = dgl.data.CoraGraphDataset()
# print(f"Number of categories: {dataset.num_classes}")
g = dataset[0]

print("Node data", g.ndata)
print("Edge data", g.edata)
print("classes: ", dataset.num_classes)
print("in_feat: ", g.ndata["feat"].shape[1])
model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)
train(g, model)
end = time.time()

print("Time Taken: ", end-start)



