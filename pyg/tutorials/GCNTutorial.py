import torch as th
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

'''
Two-Layer GCN Example Implementation
Can also be used to test my GCN made with PyG primitives
'''
class GCN(th.nn.Module):
    def __init__(self, dataset):
        super(GCN, self).__init__()

        self.conv1 = GCN_(dataset.num_node_features, 16)
        self.conv2 = GCN_(16, dataset.num_classes)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
'''
Note that the non-linearity is not integrated in the conv calls and 
hence needs to be applied afterwards (something which is consistent across all operators in PyG). 
Here, we chose to use ReLU as our intermediate non-linearity and finally output a softmax distribution over the number of classes. 
'''
datasetDirectory = "/Users/nikhi/Documents/UIUC/AdaptLab/gnn-dsl/pyg/datasets"


data = Planetoid(root=datasetDirectory, name="Cora")
model = GCN(data)
graph = data[0]
# learn what this is
optimizer = th.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(50):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], graph.y[data.train_mask]) # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
    if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f})"
        )
    loss.backward()
    optimizer.step()
# https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == graph.y[data.test_mask]).sum()
accuracy = int(correct) / int(data.test_mask.sum())

print(accuracy)