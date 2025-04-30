r"""
GNN implementations modified from two papers `"Motif-based Graph Self-Supervised Learning for Molecular Property Prediction"
    <https://arxiv.org/abs/2110.00987>`_ and `"Motif-Based Graph Representation Learning with Application to Chemical Molecules"
    <https://www.mdpi.com/2227-9709/10/1/8>`_ .
"""

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.nn import NNConv
from torch_geometric.nn.inits import zeros

num_atom_type = 120 # including the extra mask tokens
num_chirality_tag = 3
num_hybridization_type = 6

num_bond_type = 6 # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GNNNodeEncoder(torch.nn.Module):
    def __init__(self, num_layer, emb_dim,  JK="last", drop_ratio=0.2, gnn_type="gin", aggr='add'):
        super(GNNNodeEncoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        assert gnn_type in ["gin", "gcn", "gat", "graphsage", "nnconv"], "GNN type not implemented."

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embedding3 = torch.nn.Embedding(num_hybridization_type, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr=aggr))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr=aggr))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, aggr=aggr))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, aggr=aggr))
            elif gnn_type == "nnconv":
                self.gnns.append(NNConv(emb_dim, aggr=aggr, bias=True))
            else:
                raise ValueError("Invalid graph convolution type.")

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batch):
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        x = self.x_embedding1(x[:, 0].long()) + self.x_embedding2(x[:, 1].long()) + self.x_embedding3(x[:, 2].long())

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

                ### Different implementations of JK
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)
        else:
            raise ValueError("Invalid Jump knowledge.")

        return node_representation

class NodeEncodeInterface(nn.Module):
    def __init__(self, node_encoder):
        super(NodeEncodeInterface, self).__init__()
        self.node_encoder = node_encoder

    def forward(self, batch):
        batch_dict = {}

        for i, tensor_value in enumerate(batch.batch):
            scalar_value = tensor_value.item()  # Convert the tensor value to a scalar (integer)
            if scalar_value not in batch_dict:
                batch_dict[scalar_value] = []
            batch_dict[scalar_value].append(i)

        # Encode all nodes in the graph using the node_encoder
        node_representation = self.node_encoder(batch)

        return node_representation, batch_dict

class GNNGraphEncoder(torch.nn.Module):
    def __init__(self, node_encoder, emb_dim, graph_pooling="add"):
        super(GNNGraphEncoder, self).__init__()
        self.node_encoder = node_encoder
        self.emb_dim = emb_dim

        if graph_pooling in ["sum", "add"]:
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        self.readout = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, batch):
           
        mask = torch.nonzero(batch.x[:, 0] == 5.0).squeeze(1)
        node_representation = self.node_encoder(batch)
        graph_embedding = self.pool(node_representation, batch.batch)
        graph_embedding = self.readout(graph_embedding)
        return graph_embedding, node_representation[mask]



class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
       
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        #edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) # only use bond type as edge attr

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)

class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class NNConv(MessagePassing):

    """
    Reference: `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ and `"Neural Message Passing for NMR Chemical Shift Prediction"
    <https://pubs.acs.org/doi/10.1021/acs.jcim.0c00195>`_.
    """
    def __init__(self, emb_dim, aggr="add", bias=False):
        super(NNConv, self).__init__()
        self.aggr = aggr
        self.emb_dim = emb_dim

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        self.edge_nn = torch.nn.Linear(emb_dim, emb_dim * emb_dim)
        self.gru = nn.GRU(emb_dim, emb_dim)

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))
            zeros(self.bias)
        else:
            self.bias = None

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        x = self.linear(x)
        out = self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

        if self.bias is not None:
            out = out + self.bias

        _, out = self.gru(out.unsqueeze(0), x.unsqueeze(0))
        out = out.squeeze(0)

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)
        weight = weight.view(-1, self.emb_dim, self.emb_dim)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)
        self.out = torch.nn.Linear(heads * emb_dim, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_index = edge_index[0]
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        norm = self.norm(edge_index, x.size(0), x.dtype)
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        x = self.weight_linear(x)
        out = self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)
        return self.out(out)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])
        out = x_j * alpha.view(-1, self.heads, 1)
        out = out.view(-1, self.heads * self.emb_dim)
        return out


