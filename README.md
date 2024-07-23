# GNN DSL Prep Work

Simple project to understand and use Graph Neural Networks with ADAPT Lab.

## Primitive Implementations
`dgl/` contains primitive implementations of models including Graph Attention Networks (GAT), Graph Convolutional Networks (GCN), and GraphSAGE. Running the associated `...Test.py` class would train and use the model.

By primitive implementations I am constructing the specific model (GAT, GCN, etc.) with no previous base class. This way I can see the specific aggregate and update operations during each forward pass, instead of just using a pre-defined GCN object. 

`pyg/` contains a primitive implemntation of a Graph Convolutional Network.

## Papers
Understanding the material in `papers/` is important to understand the theory behind these different models, so I included it here. 

