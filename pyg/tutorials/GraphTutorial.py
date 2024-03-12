import torch as th
from torch_geometric.data import Data
# single graph is represented by a torch_geometric.data.Data instance
from torch_geometric.datasets import TUDataset, Planetoid
#To load the ENZYMES dataset (consisting of 600 graphs within 6 classes) 
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
# to visualize the graph

'''
ATTRIBUTES:
data.x: Node feature matrix with shape [num_nodes, num_node_features]
data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
 - [[0, 1, 1, 2],
    [1, 0, 2, 1]]   shows that edges exist between 0->1, 1->0, 1->2, 2->1
data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
data.pos: Node position matrix with shape [num_nodes, num_dimensions]

METHODS:
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data
'''
datasetDirectory = "/Users/nikhi/Documents/UIUC/AdaptLab/gnn-dsl/pyg/datasets"

#installing ENZYME dataset from https://chrsmrrs.github.io/datasets/docs/datasets/
enzymeDataset = TUDataset(root=datasetDirectory, name="ENZYMES")
enzymeGraph0 = enzymeDataset[0]
print(enzymeGraph0)
graph0visual = to_networkx(enzymeGraph0, to_undirected=True)
#nx.draw(graph0visual)
#plt.show()

#installing Cora, benchmark for GCN
'''
Cora (and other graphs) also have other attributes which are: train_mask, val_mask, test_mask
    train_mask denotes against which nodes to train (140 nodes),
    val_mask denotes which nodes to use for validation, e.g., to perform early stopping (500 nodes),
    test_mask denotes against which nodes to test (1000 nodes).
'''
coraDataset = Planetoid(root=datasetDirectory, name="Cora")
#cora only has one graph
coraGraph = coraDataset[0]
print(coraGraph)
coraGraphVisual = to_networkx(coraGraph, to_undirected=True)
#nx.draw(coraGraphVisual)
#plt.show()

'''
Rest of information is on: https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
Graphs can also be put through batches (making smaller graphs for parallel computing)
Graphs/Data can be transformed with a lot of different functions in torch_geometric.transforms
'''