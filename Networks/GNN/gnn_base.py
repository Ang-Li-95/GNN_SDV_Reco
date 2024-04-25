import os, sys
sys.path.append("/users/ang.li/public/SoftDV/ML/GNN_SDV_Reco/")
import torch
from torch_geometric.loader import DataLoader
from torch.nn import Linear
from Networks.utils import *
import pytorch_lightning as L
from sklearn.metrics import roc_auc_score

class GNNBase(L.LightningModule):
  def __init__(self,hparams):
    super().__init__()
    """
    Set up the hyperparameters.
    """
    self.save_hyperparameters(hparams)

  def setup(self, stage):
    """
    This prepares the data.
    stage can be 'fit', 'validate', 'test', or 'predict'.
    """
    self.trainset, self.valset, self.testset = split_dataset(input_dir=self.hparams['input_dir'],split=self.hparams['split'],seed=self.hparams['seed'])

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

  def make_input(self,batch):
    node_features = batch.x
    edges = batch.pred_edges_emb
    edges_y = batch.edges_y
    
    input_d = {
        'node_features': node_features,
        'edges': edges,
        'edges_y': edges_y,
        }
    return input_d

  def log_metrics(self, score, preds, truth, batch, loss):

      edge_positive = preds.sum().float()
      edge_true = truth.sum().float()
      edge_true_positive = (
          (truth.bool() & preds).sum().float()
      )

      eff = edge_true_positive.clone().detach() / max(1, edge_true)
      pur = edge_true_positive.clone().detach() / max(1, edge_positive)

      auc = roc_auc_score(truth.bool().cpu().detach(), score.cpu().detach())

      current_lr = self.optimizers().param_groups[0]["lr"]
      self.log_dict(
          {
              "val_loss": loss,
              "auc": auc,
              "eff": eff,
              "pur": pur,
              "current_lr": current_lr,
          }, on_epoch=True, on_step=False, batch_size=1
      )

  def training_step(self, batch, batch_idx):
    """
    This should just include the general processes during training and be flexible for different GNN models. 
    """

    # Get input and truth information
    input_d = self.make_input(batch)

    # Run network
    #pos_weight = torch.tensor((~input_d['edges_y'].bool()).sum() / input_d['edges_y'].sum() )
    pos_weight = ((~input_d['edges_y'].bool()).sum() / input_d['edges_y'].sum() ).clone().detach()
    output = self(input_d['node_features'],input_d['edges']).squeeze()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output, input_d['edges_y'].float(), weight=None, pos_weight=pos_weight
                    )

    self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=1)

    return loss

  def evaluate_network(self,batch,batch_idx, log):
    """
    This is the general procedure to evaluate the network. Will be called for validation and testing.
    """

    # Get input and truth information
    input_d = self.make_input(batch)

    # Run network
    #pos_weight = torch.tensor((~input_d['edges_y'].bool()).sum() / input_d['edges_y'].sum() )
    pos_weight = ((~input_d['edges_y'].bool()).sum() / input_d['edges_y'].sum() ).clone().detach()
    output = self(input_d['node_features'],input_d['edges']).squeeze()
    # Get the score ranging from 0 to 1
    score = torch.sigmoid(output)
    preds = score > self.hparams["edge_cut"]

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    output, input_d['edges_y'].float(), weight=None, pos_weight=pos_weight
                    )

    if log:
        self.log_metrics(score, preds, input_d['edges_y'], batch, loss)

    return {
        "loss": loss,
        "score": score,
        "preds": preds,
        "truth": input_d['edges_y'],
    }

  def validation_step(self, batch, batch_idx):
      """
      Step to evaluate the model's performance
      """

      outputs = self.evaluate_network(
          batch, batch_idx, log=True
      )

      return outputs["loss"]

  def test_step(self, batch, batch_idx):
      """
      Step to evaluate the model's performance
      """
      outputs = self.evaluate_network(
          batch, batch_idx, log=False
      )

      return outputs

  def predict_step(self, batch, batch_idx):
      """
      Step to evaluate the model's performance
      """
      outputs = self.evaluate_network(
          batch, batch_idx, log=False
      )

      return outputs
