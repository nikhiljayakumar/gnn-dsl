import torch as th
from torch import nn
from torch.nn import functional as F

import dgl
from dgl import function as fn
from dgl import DGLError
from dgl.utils import check_eq_shape, expand_as_pair

class graphSAGE(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type,
        # dropout is a pytorch function that randomly zeros out values of input to reduce
        # overfitting, when neural network works well with training data but not new data
        feat_drop = 0.0,
        bias=True,
        norm=None,
        activation=None
    ):
        super(graphSAGE, self).__init__()
        valid_aggre_types = {"mean", "gcn", "pool", "lstm"}
        if aggregator_type not in valid_aggre_types:
            raise DGLError("aggregator type not in valid types")
        
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.aggre_type = aggregator_type
        self.activation = activation
        self.bias = bias
        self.norm = norm

        self.feat_drop = nn.Dropout(feat_drop, feat_drop)
        
        if aggregator_type == "pool":
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == "lstm":
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(th.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)
    
    
        self.reset_parameters()




    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.aggre_type == "pool":
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self.aggre_type == "lstm":
            self.lstm_reset()
        if self.aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def lstm_reset(self, nodes):
        m = nodes.mailbox["m"]
        batch_size = m.shape[0]
        h = (
            m.new_zeros((1, batch_size, self._in_src_feats)),
            m.new_zeros((1, batch_size, self._in_src_feats))
        )
        _, (rst, _) = self.lstm(m, h)
        return {"neigh": rst.squeeze(0)}
    
    def forward(self, graph, feat, edge_weight=None):

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            dst_src = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")

        h_self = feat_dst

        if graph.num_edges() == 0:
            graph.dstdata["neigh"] = th.zeros(feat_dst.shape[0], self._in_src_feats).to(feat_dst)

        #linear transformatioon before message passing
        lin_before_mp = self._in_src_feats > self._out_feats

        if self.aggre_type == "mean":
            graph.srcdata["h"] = (self.fc_neigh(feat_src) if lin_before_mp else feat_src)
            graph.update_all(msg_fn, fn.mean("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)

        elif self.aggre_type == "gcn":
            check_eq_shape(feat)
            graph.srcdata["h"] = (self.fc_neigh(feat_src) if lin_before_mp else feat_src)
            if (isinstance(feat, tuple)):
                graph.dstdata["h"] = graph.srcdata["h"][: graph.num_dst_nodes()]
            else:
                if graph.is_block:
                    graph.dstdata["h"] = graph.srcdata["h"][: graph.number_of_dst_nodes()]
                else:
                    graph.dstdata["h"] = graph.srcdata["h"]
            graph.update_all(msg_fn, fn.sum("m", "neigh"))
            #divide in_degrees
            degs = graph.in_degrees().to(feat_dst)
            h_neigh = (graph.dstdata["neigh"] + graph.dstdata["h"]) / (degs.unsqueeze(-1) + 1)
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)
        elif self.aggre_type == "pool":
            graph.srcdata["h"] = F.relu(self.fc_pool(feat_src))
            graph.update_all(msg_fn, fn.max("m", "neigh"))
            h_neigh = self.fc_neigh(graph.dstdata["neigh"])
        
        if self.aggre_type == "gcn":
            rst = h_neigh

            if self.bias is not None:
                rst = rst + self.bias
        else:
            rst = self.fc_self(h_self) + h_neigh
        
        if self.activation is not None:
            rst = self.activation(rst)

        if self.norm is not None:
            rst = self.norm(rst)
        return rst

if __name__ == "__main__":
    g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    g = dgl.add_self_loop(g)
    feat = th.ones((6, 10))
    conv = graphSAGE(10, 2, 'pool')
    res = conv(g, feat)
    print(res)                                 
