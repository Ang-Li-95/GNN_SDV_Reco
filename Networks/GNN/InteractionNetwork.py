import os
import sys

sys.path.append("/users/ang.li/public/SoftDV/ML/GNN_SDV_Reco/")

#import lightning as L
import pytorch_lightning as L
import torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add, scatter_mean, scatter_max
#from Networks.utils import mlp, build_edges, split_dataset, graph_intersection
from Networks.utils import *
from Networks.GNN.gnn_base import GNNBase

class InteractionNetwork(GNNBase):
  def __init__(self, hparams):
    super().__init__(hparams)

    # This MLP encodes node features
    # Input: node features with dimension [N_nodes, hparams['input_dim']]
    # Output: [N_nodes,  hparams['node_encode_dim']]
    self.node_encoder = mlp(
        node_sizes = [hparams['input_dim']] + [hparams['hidden_dim']] * hparams['n_layers'] + [hparams['node_encode_dim']], 
        hidden_activation=hparams["activation"],
        output_activation=None,
        layer_norm=hparams["layernorm"]
        )

    # This MLP encodes edge features
    # Input: combined information from the two nodes connected by the edge [N_edges, 2*hparams['node_encode_dim']]
    # Output: [N_edges, hparams['edge_encode_dim']]
    self.edge_encoder = mlp(
        node_sizes = [2 * hparams['node_encode_dim']] + [hparams['hidden_dim']] * hparams['n_layers'] + [hparams['edge_encode_dim']],
        hidden_activation=hparams["activation"],
        output_activation=None,
        layer_norm=hparams["layernorm"]
        )

    # This MLP is used in the message passing step
    # For nodes, the message passing step collects information from the connected edges 
    # Input: original node combined with information from connected edges [N_nodes, hparams['node_encode_dim'] + hparams['edge_encode_dim']]
    # Output: [N_nodes, hparams['node_encode_dim']]
    self.node_update = mlp(
        node_sizes = [hparams['node_encode_dim'] + hparams['edge_encode_dim']] + [hparams['hidden_dim']] * hparams['n_layers'] + [hparams['node_encode_dim']],
        hidden_activation=hparams["activation"],
        output_activation=None,
        layer_norm=hparams["layernorm"]
        )

    # This MLP is used in the message passing step
    # After the nodes are updated, we collect the information of connected nodes and edge to update the edge
    # Input: original edge combined with information from connected nodes [N_edges, hparams['edge_encode_dim'] + 2*hparams['node_encode_dim']]
    # Output: [N_edges, hparams['edge_encode_dim']]
    self.edge_update = mlp(
        node_sizes = [hparams['edge_encode_dim'] + 2 * hparams['node_encode_dim']] + [hparams['hidden_dim']] * hparams['n_layers'] + [hparams['edge_encode_dim']],
        hidden_activation=hparams["activation"],
        output_activation=None,
        layer_norm=hparams["layernorm"]
        )

    # This MLP is used to predict the edge class, it should be similar with the `edge_update`
    # but outputs a single number of each edge 
    # Input: [N_edges, hparams['edge_encode_dim'] + 2*hparams['node_encode_dim']]
    # Output: [N_edges, 1]
    self.edge_classification = mlp(
        node_sizes = [hparams['edge_encode_dim'] + 2 * hparams['node_encode_dim']] + [hparams['hidden_dim']] * hparams['n_layers'] + [1],
        hidden_activation=hparams["activation"],
        output_activation=None,
        layer_norm=hparams["layernorm"]
        )

  def reset_parameters(self):
      for layer in self.modules():
          if type(layer) is Linear:
              torch.nn.init.xavier_uniform(layer.weight.data)
              layer.bias.data.fill_(0)

  def massege_passing(self,node_enc,sender,receiver,edge_enc):
    message = scatter_add(edge_enc,receiver,dim=0,dim_size=node_enc.shape[0])
    nodes_enc_update = self.node_update(torch.cat([node_enc,message],dim=-1))
    nodes_enc_update += node_enc
    edge_enc_update = self.edge_update(torch.cat([edge_enc,node_enc[sender],node_enc[receiver]],dim=-1))
    edge_enc_update += edge_enc
    return nodes_enc_update, edge_enc_update

  def output_step(self,node_enc,sender,receiver,edge_enc):
    out = self.edge_classification(torch.cat([edge_enc,node_enc[sender],node_enc[receiver]],dim=1)).squeeze(-1)
    return out

  def forward(self,nodes,edges):
    sender,receiver = edges
    # First encode node and edges
    node_enc = self.node_encoder(nodes)
    edge_enc = self.edge_encoder(torch.cat([node_enc[sender], node_enc[receiver]],dim=1))

    # Then perform the message passing
    for npass in range(self.hparams["n_message_passing"]):
      node_enc, edge_enc = self.massege_passing(node_enc, sender, receiver, edge_enc)

    # Calculate the edge classes
    output = self.output_step(node_enc,sender,receiver,edge_enc)

    return output
