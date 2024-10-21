import os
import torch
import numpy as np
import pickle
import awkward as ak
import argparse
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from networkx.algorithms.clique import find_cliques

from Networks.utils import graph_intersection

def plt_dist(distance,edge_y):
    distance_true = distance[edge_y==True]
    distance_false = distance[edge_y==False]
    fig, ax = plt.subplots()
    h_dist_true = ax.hist(distance_true,bins=100,range=(0,0.1),density=True,histtype='step',fill=False,label='True')
    h_dist_false = ax.hist(distance_false,bins=100,range=(0,0.1),density=True,histtype='step',fill=False,label='False')
    #ax.set_title(fn)
    ax.set_xlabel('EMB_distance')
    ax.set_ylabel('A.U.')
    ax.legend()
    plt.savefig(args.output_dir+'/dist.png')
    plt.close(fig)

def plt_score(score,edge_y):
    score_true = score[edge_y==True]
    score_false = score[edge_y==False]
    fig, ax = plt.subplots()
    h_dist_true = ax.hist(score_true,bins=100,range=(0,1),density=True,histtype='step',fill=False,label='True')
    h_dist_false = ax.hist(score_false,bins=100,range=(0,1),density=True,histtype='step',fill=False,label='False')
    #ax.set_title(fn)
    ax.set_xlabel('GNN_score')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(args.output_dir+'/score.png')
    plt.close(fig)

def plt_Nedges(nedges_all,nedges_matched,nedges_nomatched):
    fig, ax = plt.subplots()
    h_nedges_all = ax.hist(nedges_all,bins=20,range=(0,20),density=False,histtype='step',fill=False,label='All')
    h_nedges_true = ax.hist(nedges_matched,bins=20,range=(0,20),density=False,histtype='step',fill=False,label='Truth')
    h_nedges_False = ax.hist(nedges_nomatched,bins=20,range=(0,20),density=False,histtype='step',fill=False,label='False')
    #ax.set_title(fn)
    ax.set_xlabel('Number of edges')
    ax.set_ylabel('A.U.')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(args.output_dir+'/nedges.png')
    plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', help='Directory that includes the data to be processed [NANOAOD]')
parser.add_argument('--output_dir', help='Directory of output')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

filelist = []
for fn in os.listdir(args.input_dir):
  if fn.endswith('.pt'):
    filelist.append(os.path.join(args.input_dir,fn))

stores = []
distcut=0.02
scorecut=0.9
for fn in filelist:
  d = torch.load(fn)
  for i in range(len(d)):
      # Initiate data structure
      store = {
          "LLP_ngen": [0,0],
          "LLP_nmatch_clus": [0,0],
          "clus_ntks": [],
          "clus_LLP_idx": [],
          "clus_nmatchtks": [],
          }

      dist = d[i].distances_emb.numpy()
      ey = d[i].edges_y.numpy()
      s = d[i].score.numpy()
      # Define selections
      sel = (dist<distcut) & (s>scorecut)
      edge_idx_sel = d[i].pred_edges_emb[:,sel]
      edge_idx_sel_flip = edge_idx_sel.flip(0)
      edge_idx_sel,bidir = graph_intersection(edge_idx_sel,edge_idx_sel_flip)
      edge_idx_sel = edge_idx_sel[:,bidir]
      # Construct graph and clusters
      gtemp = Data(edge_index=edge_idx_sel)
      gnx = to_networkx(gtemp,to_undirected=True)
      cstemp = list(find_cliques(gnx))
      cs = []
      for c in cstemp:
        if len(c)>1:
          cs.append(c)

      # Get the truth graph
      gtruthtemp = Data(edge_index=d[i].true_edges)
      gtruth = to_networkx(gtruthtemp,to_undirected=True)
      tktruthtemp = list(find_cliques(gtruth))
      tkrel = set(d[i].true_edges[0])
      tktruth = []
      for tk in tktruthtemp:
        if len(tk)>1:
          tktruth.append(tk)
        elif (len(tk)==1) and (tk[0] in tkrel):
          tktruth.apend(tk)
      assert len(tktruth)<=2

      # Match cluster with truth
      for ic in cs:
        n_match = np.array([0] * len(tktruth))
        for t in range(len(tktruth)):
          for iic in ic:
            if iic in tktruth[t]:
              n_match[t] += 1
        LLP_idx = -1
        nmatchtk = 0
        if len(n_match)>0:
          LLP_idx_argmax = n_match.argmax()
          nmatchtk = n_match[LLP_idx_argmax]
          if nmatchtk>1:
            LLP_idx = LLP_idx_argmax
            store['LLP_nmatch_clus'][LLP_idx] += 1

        store['clus_ntks'].append(len(ic))
        store['clus_LLP_idx'].append(LLP_idx)
        store['clus_nmatchtks'].append(nmatchtk)

      for illp in range(len(tktruth)):
        store["LLP_ngen"][illp] = len(tktruth[illp])

      stores.append(store)
  del d

arr = ak.Array(stores)
ak.to_parquet(arr,args.output_dir+'/output.parquet')
