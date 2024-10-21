import os
import torch
import uproot
import numpy as np
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import argparse

def adddistweight(data):
  dist = data.distances_emb
  y = data.edges_y
  N_true = sum(y==True)
  N_false = sum(y==False)
  #s_false = N_false/(1/dist[y==False]).sum()
  s_false = N_false/(N_false-(dist[y==False]).sum())
  s_true = N_true/(dist[y==True]).sum()
  #w_false = s_false/dist[y==False]
  #w_true = s_true*dist[y==True]
  edge_weight = torch.ones_like(y).float()
  edge_weight[y==True] = s_true*dist[y==True]
  #edge_weight[y==False] = s_false/dist[y==False]
  edge_weight[y==False] = s_false*(1-dist[y==False])
  data.edge_weight = edge_weight
  return data

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help='Directory that includes the data to be processed')
parser.add_argument('--output_dir', help='Directory of output')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)

for s in os.listdir(args.input_dir):
  inputname = os.path.join(args.input_dir,s)

  datas = torch.load(inputname)
  datas_filtered = []
  for i in range(len(datas)):
    if (~datas[i].edges_y).sum()==0 or (datas[i].edges_y).sum()==0:
      continue
    data_new = adddistweight(datas[i])
    #data_new = datas[i]
    datas_filtered.append(data_new)
  torch.save(datas_filtered,args.output_dir+'/{}'.format(s))

