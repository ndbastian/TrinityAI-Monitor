import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv
from dgl.nn.pytorch import GraphConv
from models.GCN.graphconv_edge_w import GraphConvEdgeWeight
from models.GCN.graphconv_edge_weight import GraphConvEdgeWeight as GraphConvEdgeWeight2

# ============================================================================================================

class GCN_3_layer(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, num_classes):
        super(GCN_3_layer, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, num_classes)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConv(in_feats, h1_size)
        self.conv2 = GraphConv(h1_size, h2_size)
        self.conv3 = GraphConv(h2_size, h3_size)
        self.conv4 = GraphConv(h3_size, num_classes)

    def forward(self, g, inputs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = F.relu(self.conv3(g, out))
        out = self.conv4(g, out)
        out = out + res_op
        return out


class GCN_3_layer_edge_weight(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, num_classes):
        super(GCN_3_layer_edge_weight, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, num_classes)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConvEdgeWeight(in_feats, h1_size)
        self.conv2 = GraphConvEdgeWeight(h1_size, h2_size)
        self.conv3 = GraphConvEdgeWeight(h2_size, h3_size)
        self.conv4 = GraphConvEdgeWeight(h3_size, num_classes)

    def forward(self, g, inputs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = F.relu(self.conv3(g, out))
        out = self.conv4(g, out)
        out = out + res_op
        return out


class GCN_3_layer_fc(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, num_classes):
        super(GCN_3_layer_fc, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h3_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConv(in_feats, h1_size)
        self.conv2 = GraphConv(h1_size, h2_size)
        self.conv3 = GraphConv(h2_size, h3_size)
        self.op = nn.Linear(h3_size, num_classes)

    def forward(self, g, inputs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = self.conv3(g, out)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out


class GCN_3_layer_edge_weight_fc(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, num_classes):
        super(GCN_3_layer_edge_weight_fc, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h3_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConvEdgeWeight(in_feats, h1_size)
        self.conv2 = GraphConvEdgeWeight(h1_size, h2_size)
        self.conv3 = GraphConvEdgeWeight(h2_size, h3_size)
        self.op = nn.Linear(h3_size, num_classes)

    def forward(self, g, inputs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = self.conv3(g, out)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out


# ============================================================================================================
class GCN_4_layer(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, h4_size, num_classes):
        super(GCN_4_layer, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, num_classes)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConv(in_feats, h1_size)
        self.conv2 = GraphConv(h1_size, h2_size)
        self.conv3 = GraphConv(h2_size, h3_size)
        self.conv4 = GraphConv(h3_size, h4_size)
        self.conv5 = GraphConv(h4_size, num_classes)

    def forward(self, g, inputs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = F.relu(self.conv3(g, out))
        out = F.relu(self.conv4(g, out))
        out = self.conv5(g, out)
        out = out + res_op

        return out


class GCN_4_layer_edge_weight(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, h4_size, num_classes):
        super(GCN_4_layer_edge_weight, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, num_classes)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConvEdgeWeight(in_feats, h1_size)
        self.conv2 = GraphConvEdgeWeight(h1_size, h2_size)
        self.conv3 = GraphConvEdgeWeight(h2_size, h3_size)
        self.conv4 = GraphConvEdgeWeight(h3_size, h4_size)
        self.conv5 = GraphConvEdgeWeight(h4_size, num_classes)

    def forward(self, g, inputs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = F.relu(self.conv3(g, out))
        out = F.relu(self.conv4(g, out))
        out = self.conv5(g, out)
        out = out + res_op
        return out


class GCN_4_layer_fc(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, h4_size, num_classes):
        super(GCN_4_layer_fc, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h4_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConv(in_feats, h1_size)
        self.conv2 = GraphConv(h1_size, h2_size)
        self.conv3 = GraphConv(h2_size, h3_size)
        self.conv4 = GraphConv(h3_size, h4_size)
        self.op = nn.Linear(h4_size, num_classes)

    def forward(self, g, inputs,*args,**kwargs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = F.relu(self.conv3(g, out))
        out = self.conv4(g, out)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out


class GCN_4_layer_edge_weight_fc2(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, h4_size, num_classes):
        super().__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h4_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConv(in_feats, h1_size, weight=True, bias=True)
        self.conv2 = GraphConv(h1_size, h2_size , weight=True, bias=True)
        self.conv3 = GraphConv(h2_size, h3_size , weight=True, bias=True)
        self.conv4 = GraphConv(h3_size, h4_size, weight=True, bias=True)
        self.op = nn.Linear(h4_size, num_classes)

    def forward(self, g, inputs , edge_weights):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs , edge_weight=edge_weights))
        out = F.relu(self.conv2(g, out ,  edge_weight=edge_weights))
        out = F.relu(self.conv3(g, out ,  edge_weight=edge_weights))
        out = self.conv4(g, out ,  edge_weight=edge_weights)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out


class GCN_4_layer_edge_weight_fc3(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, h4_size, num_classes):
        super().__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h4_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConvEdgeWeight2(in_feats, h1_size)
        self.conv2 = GraphConvEdgeWeight2(h1_size, h2_size)
        self.conv3 = GraphConvEdgeWeight2(h2_size, h3_size)
        self.conv4 = GraphConvEdgeWeight2(h3_size, h4_size)
        self.op = nn.Linear(h4_size, num_classes)

    def forward(self, g, inputs , edge_weight):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs , edge_weights=edge_weight))
        out = F.relu(self.conv2(g, out ,  edge_weights=edge_weight))
        out = F.relu(self.conv3(g, out ,  edge_weights=edge_weight))
        out = self.conv4(g, out ,  edge_weights=edge_weight)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out


class GCN_4_layer_edge_weight_fc(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, h4_size, num_classes):
        super(GCN_4_layer_edge_weight_fc, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h4_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConvEdgeWeight(in_feats, h1_size)
        self.conv2 = GraphConvEdgeWeight(h1_size, h2_size)
        self.conv3 = GraphConvEdgeWeight(h2_size, h3_size)
        self.conv4 = GraphConvEdgeWeight(h3_size, h4_size)
        self.op = nn.Linear(h4_size, num_classes)

    def forward(self, g, inputs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = F.relu(self.conv2(g, out))
        out = F.relu(self.conv3(g, out))
        out = self.conv4(g, out)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out


class MLP_4_layer(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, h3_size, h4_size, num_classes):
        super(MLP_4_layer, self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h4_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.lin1 = nn.Linear(in_feats, h1_size)
        self.lin2 = nn.Linear(h1_size, h2_size)
        self.lin3 = nn.Linear(h2_size, h3_size)
        self.lin4 = nn.Linear(h3_size, h4_size)
        self.op = nn.Linear(h4_size, num_classes)

    def forward(self, g, inputs,*args,**kwargs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.lin1(inputs))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out



class GCN_2_layer_fc(nn.Module):
    def __init__(self, in_feats, h1_size, h2_size, num_classes):
        super(GCN_2_layer_fc,self).__init__()

        # linear layer to match feature dimension
        self.residual = nn.Linear(in_feats, h2_size)
        # nn.Conv1d(in_feats,out_channels,kernel_size=1,stride=(stride, 1))

        self.conv1 = GraphConv(in_feats, h1_size)
        self.conv2 = GraphConv(h1_size, h2_size)
        self.op = nn.Linear(h2_size, num_classes)

    def forward(self, g, inputs,*args,**kwargs):
        # residual
        res_op = self.residual(inputs)

        out = F.relu(self.conv1(g, inputs))
        out = self.conv2(g, out)
        out = F.relu(out + res_op)
        out = self.op(out)
        return out


class GCN_1_layer_fc(nn.Module):
    def __init__(self, in_feats, h1_size, num_classes):
        super(GCN_1_layer_fc,self).__init__()

        self.conv1 = GraphConv(in_feats, h1_size)
        self.op = nn.Linear(h1_size, num_classes)

    def forward(self, g, inputs,*args,**kwargs):
        out = F.relu(self.conv1(g, inputs))
        out = self.op(out)
        return out

##https://github.com/ZeroRin/BertGCN/blob/main/model/torch_gat.py
class GAT(nn.Module):
    """
    Graph Attention Networks in DGL using SPMV optimization.
    References
    ----------
    Paper: https://arxiv.org/abs/1710.10903
    Author's code: https://github.com/PetarV-/GAT
    Pytorch implementation: https://github.com/Diego999/pyGAT
    """
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs,*args,**kwargs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits


from models.graph_transformer_net import graph_model
gcn_models = {
        "gcn3layer": GCN_3_layer,
        "gcn3layer_edge": GCN_3_layer_edge_weight,
        "gcn3layer_fc": GCN_3_layer_fc,
        "gcn3layer_edge_fc": GCN_3_layer_edge_weight_fc,
        "gcn4layer": GCN_4_layer,
        "gcn4layer_edge": GCN_4_layer_edge_weight,
        "gcn4layer_fc": GCN_4_layer_fc,
        "gcn4layer_edge_fc": GCN_4_layer_edge_weight_fc,
        "gcn4layer_edge_fc_mine": GCN_4_layer_edge_weight_fc2,
        "gcn4layer_edge_fc_mine_better": GCN_4_layer_edge_weight_fc3,
        "mlp4layer":MLP_4_layer,
        "gcn2layer_fc":GCN_2_layer_fc,
        "gcn1layer_fc":GCN_1_layer_fc,
        "gat":GAT,
        "transformer": graph_model,
         }
