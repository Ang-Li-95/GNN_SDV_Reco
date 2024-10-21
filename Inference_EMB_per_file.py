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

from Networks.Embedding.embedding import Embedding
from Networks.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', help='Directory that includes the data to be processed [NANOAOD]')
parser.add_argument('--output_dir', help='Directory of output')
parser.add_argument('--model', help='path of the model')
args = parser.parse_args()

# Load config file
with open('emb_config.yaml') as c:
    cl = yaml.load(c, Loader=yaml.FullLoader)
    config = cl['metric_learning_configs']

model = Embedding(config)

os.makedirs(args.output_dir, exist_ok=True)

d = load_dataset(args.input_file)
dl = DataLoader(d,batch_size=1)
trainer = Trainer()
#res = trainer.predict(model,dl,ckpt_path='EMB_out_0502_2/model-epoch=57-val_loss=0.00285.ckpt')
res = trainer.predict(model,dl,ckpt_path=args.model)

for i in range(len(d)):
    d[i].pred_edges_emb = res[i]['preds']
    d[i].distances_emb = res[i]['distances']
    d[i].edges_y = res[i]['truth']

torch.save(d,os.path.join(args.output_dir,'{}.pt'.format(os.path.splitext(os.path.basename(args.input_file))[0])))

