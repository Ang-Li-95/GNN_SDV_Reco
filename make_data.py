import os
import torch
import uproot
import numpy as np
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import argparse

def make_edges(nodes):
    if len(nodes)<2:
        return np.array([[],[]])
    l = len(nodes)
    sender = []
    receiver = []
    for i in range(l):
        for j in range(l):
            if i==j:
                continue
            sender.append(nodes[i])
            receiver.append(nodes[j])
    #print(numpy.array([sender,receiver]))
    return np.array([sender,receiver])

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help='Directory that includes the data to be processed [NANOAOD]')
parser.add_argument('--output_dir', help='Directory of output')
args = parser.parse_args()

input_features = ['SDVTrack_pt', 'SDVTrack_eta', 'SDVTrack_phi', 'SDVTrack_dxy', 'SDVTrack_dxyError', 'SDVTrack_dz', 'SDVTrack_dzError', 'SDVTrack_pfRelIso03_all', 'SDVTrack_ptError', 'SDVTrack_normalizedChi2', 'SDVTrack_numberOfValidHits']

if not os.path.exists(args.output_dir):
  os.makedirs(args.output_dir)
if not os.path.exists(os.path.join(args.output_dir,'train')):
  os.makedirs(os.path.join(args.output_dir,'train'))
if not os.path.exists(os.path.join(args.output_dir,'test')):
  os.makedirs(os.path.join(args.output_dir,'test'))

for s in os.listdir(args.input_dir):
  inputname = os.path.join(args.input_dir,s,'output')

  #f = uproot.open(inputname+"/*.root")
  #t = f["Events"]
  #arrs = t.arrays(input_features+['SDVTrack_LLPIdx'], library='np')
  
  datas = []
  geninfo = 'SDVTrack_GenSecVtxIdx'
  for arrs in uproot.iterate(inputname+"/*.root:Events",input_features+[geninfo], library='np'):
    # For each event
    for i in range(len(arrs[geninfo])):
      features = [arrs[s][i] for s in input_features] 
      testx = np.array(features).T
      #testx = np.array([arrs['SDVTrack_pt'][i],arrs['SDVTrack_eta'][i]]).T
      edges = [np.array([[],[]])]
      temp = arrs[geninfo][i]
      if len(temp)==0:
        continue
      ws = []
      for ivtx in range(max(temp)+1):
        w = np.where(temp==ivtx)[0]
        ws.append(w)
        if len(ws[-1])>1:
          edges.append(make_edges(ws[-1]))
      edges = np.concatenate(edges,axis=1)
      if edges.shape[1]==0:
          continue
      if np.isnan(testx).any() or np.isnan(edges).any():
          print("Found NAN!")
          continue
      testx = torch.from_numpy(testx)
      edges = torch.from_numpy(edges).long()
      a = Data(x=testx)
      a.true_edges = edges
      datas.append(a)
  
  datas_train, datas_test = torch.utils.data.random_split(datas,[0.8,0.2])
  torch.save(datas_train,args.output_dir+'/train/{}.pt'.format(s))
  torch.save(datas_test,args.output_dir+'/test/{}.pt'.format(s))
