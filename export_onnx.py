#!/usr/bin/env python
# coding: utf-8

import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

import onnx
import onnxruntime

from lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch
from torch_geometric.loader import DataLoader

from Networks.Embedding.embedding import Embedding
from Networks.GNN.InteractionNetwork import InteractionNetwork
from Networks.utils import *


# ### Export Embedding model

d = torch.load("Datasets/stop_M600_588_ct200_2018.pt")
#model = Embedding.load_from_checkpoint("EMB_out_0502_2/model-epoch=57-val_loss=0.00285.ckpt")
model = Embedding.load_from_checkpoint("EMB_0603_3/model-epoch=105-val_loss=0.00292.ckpt")
input_sample = d[0].x.float()
model.to_onnx('EMB_0603_3.onnx', input_sample, input_names=['input_tk'], output_names=['emb'], export_params=True,opset_version=12,dynamic_axes={"input_tk":[0]})


# ### Export GNN model

dgnn = torch.load("EMB_out_0502_2_output_rtest0p15_filtered_newweight/train/stop_M600_588_ct200_2018.pt")
input_sample = tuple((dgnn[0].x.float(),dgnn[0].pred_edges_emb))
model = InteractionNetwork.load_from_checkpoint("../GNN_SDV_Reco_localtest/GNN_dist0p15_0509/model-epoch=198-val_loss=0.45724.ckpt")
model.to_onnx('GNN_1.onnx', input_sample, input_names=['input_tk','input_edges'], output_names=['gnn'], export_params=True,opset_version=12,dynamic_axes={"input_tk":[0],"input_edges":[1]})
