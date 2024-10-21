import os
import sys

sys.path.append("/users/ang.li/public/SoftDV/ML/GNN_SDV_Reco/")

import lightning as L
#import pytorch_lightning as L
import torch
from torch_geometric.loader import DataLoader
#from Networks.utils import mlp, build_edges, split_dataset, graph_intersection
from Networks.utils import *

class Embedding(L.LightningModule):
  def __init__(self,hparams):
    super().__init__()
    """
    Set up the hyperparameters and define the architecture.
    """
    node_sizes = [hparams['input_dim']] + [hparams['hidden_dim']] * hparams['n_layers'] + [hparams['emb_dim']]
    self.network = mlp(node_sizes = node_sizes, hidden_activation=hparams["activation"],output_activation=None,layer_norm=True)

    self.save_hyperparameters(hparams)

  def setup(self, stage):
    """
    This prepares the data.
    stage can be 'fit', 'validate', 'test', or 'predict'.
    """
    self.trainset, self.valset, self.testset = split_dataset(input_dir=self.hparams['input_dir'],split=self.hparams['split'],seed=self.hparams['seed'],max_events_per_source=self.hparams['max_events_per_source'])

  def train_dataloader(self):
      if len(self.trainset) > 0:
          return DataLoader(self.trainset, batch_size=1, num_workers=16)
      else:
          return None

  def val_dataloader(self):
      if len(self.valset) > 0:
          return DataLoader(self.valset, batch_size=1, num_workers=16)
      else:
          return None

  def test_dataloader(self):
      if len(self.testset):
          return DataLoader(self.testset, batch_size=1, num_workers=16)
      else:
          return None

  def configure_optimizers(self):
      optimizer = [
          torch.optim.AdamW(
              self.parameters(),
              lr=(self.hparams["lr"]),
              betas=(0.9, 0.999),
              eps=1e-08,
              amsgrad=True,
          )
      ]
      scheduler = [
          {
              "scheduler": torch.optim.lr_scheduler.StepLR(
                  optimizer[0],
                  step_size=self.hparams["patience"],
                  gamma=self.hparams["factor"],
              ),
              "interval": "epoch",
              "frequency": 1,
          }
      ]
      return optimizer, scheduler

  def forward(self, x):
    """
    The forward method for pytorch modules. Will be used when making prediction.
    """
    scale_mean = torch.tensor([1.6466, -0.0012, -0.0039, -0.1067, 0.0449, -0.0283, 0.1674, 12.7531, 0.0178, 1.1603, 14.7395])
    scale_std = torch.tensor([3.4727, 0.7561, 1.8255, 4.6112, 0.0789, 10.1196, 0.5957, 56.4042, 0.0412, 0.7702, 4.3868])
    scale_x = (x-scale_mean)/scale_std
    out = self.network(scale_x)
    
    return out

  def training_step(self, batch, batch_idx):
    """
    This step trains the network
    """
    x = batch.x.float()
    emb_out = self(x)
    emb_out = emb_out.to(self.device)

    # Calculate the Hinge loss
    # Initiate an empty edge list
    edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)
    
    # Select query nodes
    query_indices, query = self.get_query_points(batch,emb_out)

    # For each query nodes, build edges between the node and all nodes 
    edges = self.hnm_edges(emb_out,query,query_indices,edges)

    # Now we build the truth graph, since the true edges in the input is already bi-directional, nothing needs to be done
    #true_edges = torch.cat([batch.true_edges,batch.true_edges.flip(0)], axis=-1)
    true_edges = batch.true_edges
    edges, edge_labels = graph_intersection(edges, true_edges)

    # Append truth edges to the edge list
    # FIXME: I think it would be good to remove duplicated edges
    edges, edge_labels = self.append_true_edges(edges,edge_labels,true_edges)

    # Rerun the network on all nodes included in edges
    # FIXME: But is this really needed?
    node_include = edges.unique()
    emb_out[node_include] = self(x[node_include])
    emb_out = emb_out.to(self.device)

    hinge, d = self.get_hinge_distance(emb_out,edges,edge_labels)

    if hinge[hinge == -1].shape[0]==0:
      negative_loss = torch.tensor([0])
    else:
      negative_loss = torch.nn.functional.hinge_embedding_loss(
          d[hinge == -1],
          hinge[hinge == -1],
          margin=self.hparams["margin"]**2,
          reduction="mean",
      )

    if d[hinge == 1].shape[0]==0:
      positive_loss = torch.tensor([0])
    else:
      positive_loss = torch.nn.functional.hinge_embedding_loss(
          d[hinge == 1],
          hinge[hinge == 1],
          margin=self.hparams["margin"]**2,
          reduction="mean",
      )

    loss = negative_loss + self.hparams["weight"] * positive_loss

    # Below calculates the loss similar with val
    # Calculate the Hinge loss
    # For each query nodes, build edges between the node and all nodes 
    edges_val = build_edges(emb_out, emb_out, indices=None, r_max=self.hparams['r_max'], k_max=self.hparams['k_max'])

    # Now we build the truth graph
    #true_edges = torch.cat([batch.true_edges,batch.true_edges.flip(0)], axis=-1)
    edges_val, edge_labels_val = graph_intersection(edges_val, true_edges)

    hinge_val, d_val = self.get_hinge_distance(emb_out,edges_val,edge_labels_val)

    if hinge_val.shape[0]==0:
      loss_val = torch.tensor([0])
    else:
      loss_val = torch.nn.functional.hinge_embedding_loss(
          d_val,
          hinge_val,
          margin=self.hparams["margin"]**2,
          reduction="mean",
      )

    #self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=1,prog_bar=True)
    self.log_dict({"train_loss": loss, "train_loss_val": loss_val}, on_epoch=True, on_step=False, batch_size=1,prog_bar=True, sync_dist=True)

    return loss


  def hnm_edges(self,latent,query,query_indices,edges):
    """
    This method constructs edges between nodes using Hard Negative Mining
    For each query node, it searches for nother nodes that are close to 
    the query node in the latend space, then build edges between the query 
    node and other close nodes.
    """
    new_edges = build_edges(query, latent, query_indices, self.hparams['r_max'], self.hparams['k_max'])
    edges = torch.cat([edges, new_edges], axis=-1)
    return edges

  def append_true_edges(self,edges,edge_labels,truth_edges):
    """
    This method append the truth edges to the edge list
    Doing this ensures that all truth edges are included 
    when calculating the loss function, which aimes to
    minimize the distance between nodes connected by
    truth edges in the latent space
    """
    # Append edges
    edges = torch.cat([edges.to(self.device),truth_edges.to(self.device)], axis=-1)
    # Append edge labels
    edge_labels = torch.cat([edge_labels.int().to(self.device), torch.ones(truth_edges.shape[1]).to(self.device)])
    return edges, edge_labels

  def get_query_points(self,batch,latent):
    if "all" in self.hparams['query_points']:
      query_indices = torch.arange(len(latent)).to(latent.device)
    elif "noise" in self.hparams['query_points']:
      query_indices = torch.cat(
          [torch.where(batch.is_signal==0)[0],batch.true_edges.unique()])
    else: # else we take signal edges
      query_indices = batch.true_edges.unique()

    query_indices = query_indices[torch.randperm(len(query_indices))][
        : self.hparams["points_per_batch"]
    ]
    query = latent[query_indices]

    return query_indices, query

  def get_hinge_distance(self,emb,edges,edge_labels):
    hinge = edge_labels.float().to(self.device)
    hinge[hinge==0] = -1

    sender_nodes = emb.index_select(0,edges[0]).to(self.device)
    receiver_nodes = emb.index_select(0,edges[1]).to(self.device)
    d = torch.sum((sender_nodes-receiver_nodes)**2,dim=-1)

    return hinge, d

  def evaluate_network(self, batch, batch_idx, r_max, k_max, log=False, verbose=False):
    """
    This evaluates the network, will be used in the validation and testing
    """
    x = batch.x.float()
    emb_out = self(x)
    emb_out = emb_out.to(self.device)

    # Calculate the Hinge loss
    # For each query nodes, build edges between the node and all nodes 
    edges = build_edges(emb_out, emb_out, indices=None, r_max = r_max, k_max = k_max)

    # Now we build the truth graph
    #true_edges = torch.cat([batch.true_edges,batch.true_edges.flip(0)], axis=-1)
    true_edges = batch.true_edges
    edges, edge_labels = graph_intersection(edges, true_edges)

    hinge, d = self.get_hinge_distance(emb_out,edges,edge_labels)

    if hinge.shape[0]==0:
      loss = torch.tensor([0])
    else:
      loss = torch.nn.functional.hinge_embedding_loss(
          d,
          hinge,
          margin=self.hparams["margin"]**2,
          reduction="mean",
      )
    #if torch.isnan(loss).any():
    #  print("NAN in loss")
    #  print("d")
    #  print(d)
    #  print("hinge")
    #  print(hinge)

    if hinge[hinge == -1].shape[0]==0:
      negative_loss = torch.tensor([0])
    else:
      negative_loss = torch.nn.functional.hinge_embedding_loss(
          d[hinge == -1],
          hinge[hinge == -1],
          margin=self.hparams["margin"]**2,
          reduction="mean",
      )
    if d[hinge == 1].shape[0]==0:
      positive_loss = torch.tensor([0])
    else:
      positive_loss = torch.nn.functional.hinge_embedding_loss(
          d[hinge == 1],
          hinge[hinge == 1],
          margin=self.hparams["margin"]**2,
          reduction="mean",
      )
    loss_nom = negative_loss + self.hparams["weight"] * positive_loss

    n_true_edges = true_edges.shape[1]
    n_true_positives = edge_labels.sum()
    n_edges_pred = len(edges[0])

    eff = n_true_positives / n_true_edges
    pur = n_true_positives / n_edges_pred

    if log:
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {"val_loss": loss, "val_loss_nom": loss_nom, "eff": eff, "pur": pur, "current_lr": current_lr},
            on_epoch=True,
            on_step=False,
            batch_size=1,
            prog_bar=False,
            sync_dist=True,
        )

    if verbose:
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))

    return {
        "loss": loss,
        "distances": d,
        "preds": edges,
        "truth": edge_labels,
        "truth_graph": true_edges,
        'eff': eff,
        'pur': pur
    }

  def validation_step(self, batch, batch_idx):
      """
      Step to evaluate the model's performance
      """

      outputs = self.evaluate_network(
          batch, batch_idx, self.hparams["r_val"], 150, log=True, verbose=False
      )

      return outputs["loss"]

  def test_step(self, batch, batch_idx):
      """
      Step to evaluate the model's performance
      """
      outputs = self.evaluate_network(
          batch, batch_idx, self.hparams["r_test"], 1000, log=False
      )

  def predict_step(self, batch, batch_idx):
      """
      Step to evaluate the model's performance
      """
      outputs = self.evaluate_network(
          batch, batch_idx, self.hparams["r_test"], 1000, log=False
      )

      return outputs
