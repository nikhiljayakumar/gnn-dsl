import torch as th
from torch import nn

from dgl import function as fn
from dgl import graph, add_self_loop
from dgl import DGLError
from dgl.utils import expand_as_pair
from dgl.ops import edge_softmax
from dgl.nn.pytorch.utils import Identity

class GAT(nn.Module):
    # Graph Attention Network Implementation Using DGL Primitives

    ''' 
    Aside from normal GNNs, GATs apply an attention mechanism that computes the importance of a node to another node
     - Attention mechanism: As described in the paper, the attention mechanism being impelemented is a
       single-layer feed forward neural network
     - LeakyRELU: activation function that basically disregards negative numbers
    This is normalized (using softmax) then applied to the normal forward function 
    Multi-head attention is when multiple attention mechanisms are running on the same nodes, and are 
    (oncatened unless final layer then averaged - CURRENTLY NOT IMPLEMENTED
    '''
    
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0,
        attn_drop=0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True
    ):
        super(GAT, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            '''
            nn.Linear(x,y) is single layer feed forward network with x inputs and y outputs
            if source and destination sizes are different, create different networks
            '''
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False) 
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)

        # different attention learnable parameters
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))

        '''
        Random elements are dropped (zeroed) during the training process
        Effective technique for regularization and co-adaptation of neurons
        '''
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leakyRELU = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False

        # not sure what this is
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats * num_heads)
                self.has_linear_res = True
            else:
                self.res_fc = Identity() # some placeholder
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        '''
        Re-initalizes learnable parameters
        Fc weights are initialized using Glorot uniform initializiation
        Attention weights are initialized using Xavier initialization 
        '''
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)

        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    # indegrees -> calculate the number of arrows pointing towards the node
                    raise DGLError("you specified don't allow zeros and there are zeros in the degrees")
            
            if isinstance(feat, tuple):
                # seperate src and dst features
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[0].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])

                # Multiply by the Weight Matrix 
                if not hasattr(self, "fc_src"): # if src and dst are seperate
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src.view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst.view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                # src and dst are same shape
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                    h_dst = h_dst[: graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) +  dst_prefix_shape[1:]

            '''
            Acutal Computations: Paper first concatenates Wh_i and Wh_j, but DGL first projects by
                                 a^T first (a_l*Wh_i)+(a_r*Wh_j)
            '''
            #    (  Wh_i   *   a_l    )
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_src * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, basically perform main equation
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            # activation for attention mechanism
            e = self.leakyRELU(graph.edata.pop("e"))
            # normalization (softmax) of edge weights
            graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))
            if edge_weight is not None:
                # what does this do
                graph.edata["a"] = graph.edata["a"] * edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)
            # message passing (same as normal gnn)
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual (not sure how this works)
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst += self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation 
            if self.activation:
                rst = self.activation(rst)
            
            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


if __name__ == "__main__":
    g = graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    g = add_self_loop(g)
    feat = th.ones(6, 10)
    gatconv = GAT(10, 2, num_heads=3)
    res = gatconv.forward(g, feat)
    print(res)