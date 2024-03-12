import torch

from dgl import function as fn
from dgl import DGLError
from dgl import block_to_graph
from dgl.heterograph import DGLBlock
from dgl import reverse

class MyEdgeWeightNorm(torch.nn.Module):
    def __init__(self, norm="both", eps=0.0):
        self._norm = norm
        self._eps = eps


    def forward(self, graph, edge_weight):
        
        with graph.local_scope(): #if in-place operation, affects original graph, otherwise no
            if isinstance(graph, DGLBlock): #if its a block, make it a graph
                graph = block_to_graph(graph)
            if len(edge_weight.shape) > 1:
                raise DGLError("can't use vectors as edge weights, have to be scalars")
            if self._norm == "both" and torch.any(edge_weight < 0).item():
                raise DGLError("can't have negative weights with norm='both' b/c of the way formula is defined")

            dev = graph.device # device where torch tensor is allocated (cpu in my case)
            dtype = edge_weight.dtype
            graph.srcdata["_src_out_w"] = torch.ones(
                graph.number_of_src_nodes(), dtype=dtype, device=dev
            )
            graph.dstdata["_dst_in_w"] = torch.ones(
                graph.number_of_dst_nodes(), dtype=dtype, device=dev
            )
            graph.edata["_edge_w"] = edge_weight
            # setting the source, dst as ones and edge weights as parameter
            if self._norm == "both":
                reversed_g = reverse(graph)
                reversed_g.edata["_edge_w"] = edge_weight
                # actually applying the formula
                reversed_g.update_all(
                    # message function (for the edges) and the reduce function (aggregates messages by sum for the nodes)
                    fn.copy_e("_edge_w", "m"), fn.sum("m", "out_weight")
                )
                degs = reversed_g.dstdata["out_weight"] + self._eps
                norm = torch.pow(degs, -0.5)
                graph.srcdata["_src_out_w"] = norm
            # all operations here are out-of-place
            if self._norm != "none":
                graph.update_all(
                    fn.copy_e("_edge_w", "m"), fn.sum("m", "in_weight")
                )
                degs = graph.dstdata["in_weight"] + self._eps
                if self._norm == "both":
                    #more of the formula
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                graph.dstdata["_dst_in_w"] = norm

            graph.apply_edges(
                lambda e: {
                    "_norm_edge_weights": e.src["_src_out_w"]
                    * e.dst["_dst_in_w"]
                    * e.data["_edge_w"]
                }
            )
            return graph.edata["_norm_edge_weights"]

