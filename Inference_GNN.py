#!/usr/bin/env python
# coding: utf-8
import sys
import os
import yaml
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

from lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch
from torch_geometric.loader import DataLoader

from Networks.GNN.InteractionNetwork import InteractionNetwork
from Networks.utils import *

# Load config file
with open('gnn_config.yaml') as c:
    cl = yaml.load(c, Loader=yaml.FullLoader)
    config = cl['gnn_learning_configs']

model = InteractionNetwork(config)

testdir = 'EMB_out_0502_2_output/test'
outputdir = 'GNN_out_0509_dist0p15_output'
os.makedirs(outputdir, exist_ok=True)

for sf in os.listdir(testdir):
  d = load_dataset(testdir+'/'+sf)
  dl = DataLoader(d,batch_size=1)
  trainer = Trainer()
  #res = trainer.predict(model,dl,ckpt_path='GNN_out_0507_1/model-epoch=145-val_loss=0.39244.ckpt')
  res = trainer.predict(model,dl,ckpt_path='../GNN_SDV_Reco_localtest/GNN_dist0p15_0509/model-epoch=198-val_loss=0.45724.ckpt')
  
  #print(res[0]['truth'].shape)
  
  for i in range(len(d)):
      d[i].pred_edges = res[i]['preds']
      d[i].score = res[i]['score']
      d[i].edges_y = res[i]['truth']
  
  torch.save(d,os.path.join(outputdir,sf))
  
