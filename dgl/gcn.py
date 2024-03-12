import torch

from dgl import function as fn
from dgl import DGLError
from dgl import block_to_graph
from dgl.heterograph import DGLBlock
from dgl import reverse
from dgl.utils import expand_as_pair

class MyGraphConv(torch.nn.Module):
    # one graphconv layer, defined by the main formula in GCN paper
    # edge weights are an optional addition, implemented my own in the normalizeEdgeWeights.py file

    def __init__(
        self,
        input_features,
        output_features,
        norm = "None",
        weight = True,
        bias = True,
        activation = None,
        zeros_in_degrees = False
    ):
        super(MyGraphConv, self).__init__()
        if norm not in ("None", "both", "right", "left"):
            raise DGLError("normalization must be right, left, both, or none")
        
        self._input_features = input_features
        self._output_features = output_features
        self._norm = norm
        self._activation = activation
        self._allow_zeros = zeros_in_degrees

        # use the weight matrix as the learnable parameter, so that tensors are updated during forward method
        # if weight=False, then there is no learnable parameter
        if weight:
            self._weight = torch.nn.Parameter(torch.Tensor(input_features, output_features))
        else:
            self.register_parameter("weight", None)
        
        # same for bias
        if bias:
            self._bias = torch.nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter("bias", None)

        self.reIntializeParameters()

    def reIntializeParameters(self):
        if self._weight is not None:
            torch.nn.init.xavier_uniform_(self._weight) # figure this out
        if self._bias is not None:
            torch.nn.init.zeros_(self._bias) #assuming these are in-place methods
    
    def reset_parameters(self):
        if self._weight is not None:
            torch.nn.init.xavier_uniform_(self._weight)
        if self._bias is not None:
            torch.nn.init.zeros_(self._bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zeros:
                if (graph.in_degrees() == 0).any():
                    # indegrees -> calculate the number of arrows pointing towards the node
                    raise DGLError("you specified don't allow zeros and there are zeros in the degrees")
            aggregate_fn = fn.copy_u("h", "m") # these fn methods are in place, i.e. copy graph["h"] into "m"
            
            if edge_weight is not None:
                if (edge_weight.shape[0] != graph.num_edges()):
                    raise DGLError("columns in edge_weight matrix must match number of edges")
                graph.edata["_edge_weight"] = edge_weight
                # if edge weight matrix exists, change the agregation function such that it 
                # performs elemement-wise multiplication between features of i, j
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")


            # (BarclayII) figure out what this is
            # do left or both normalization first, then do right or both (again) normalization
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                # normalization
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm
              
            if weight is not None:
                if self.weight == False:
                    raise DGLError('''external weight is provided but GraphConv was 
                                   instatiated assuming no weight, change parameter in class instatiaion''')
            else:
                weight = self._weight
            
            # ACTUAL CALCULATION PART
            
            if self._input_features > self._output_features:
                # mult by weight matrix to reduce feature size for aggregation
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum("m", "h"))
                rst = graph.dstdata["h"]
            else:
                # aggregate first then multiply by W
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum("m", "h"))
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = torch.matmul(rst, weight)
                

            # right normalization
            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                
                shape = norm.shape + (1,) * (feat_dst.dim()-1)
                norm = norm.reshape(shape)
                rst = rst * norm

            if self._bias is not None:
                rst += self._bias
            
            if self._activation is not None:
                rst = self._activation(rst)

            return rst