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

fileNames = []
for root, dirs, files in os.walk(args.input_dir):
  for ifile in files:
    if (ifile.endswith('.root')):
      fileNames.append(os.path.join(root,ifile)+':Events') 
  
datas = []
for arrs in uproot.iterate(fileNames,input_features+['SDVTrack_LLPIdx'], library='np'):
  for i in range(len(arrs['SDVTrack_LLPIdx'])):
    features = [arrs[s][i] for s in input_features] 
    testx = np.array(features).T
    #testx = np.array([arrs['SDVTrack_pt'][i],arrs['SDVTrack_eta'][i]]).T
    edges = [np.array([[],[]])]
    temp = arrs['SDVTrack_LLPIdx'][i]
    w0 = np.where(temp==0)[0]
    w1 = np.where(temp==1)[0]
    if len(w0)>1:
        edges.append(make_edges(w0))
    if len(w1)>1:
        edges.append(make_edges(w1))
    edges = np.concatenate(edges,axis=1)
    #if edges.shape[1]==0:
    #    continue
    #if np.isnan(testx).any() or np.isnan(edges).any():
    #    print("Found NAN!")
    #    continue
    testx = torch.from_numpy(testx)
    edges = torch.from_numpy(edges).long()
    a = Data(x=testx)
    a.true_edges = edges
    datas.append(a)

torch.save(datas,args.output_dir+'/MLInput.pt')
