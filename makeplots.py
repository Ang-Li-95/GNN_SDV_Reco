import os
import torch
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt

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

distcut=0.02
scorecut=0.9
distance = []
score = []
edge_y = []
nedges = []
nedges_matched = []
nedges_nomatched = []
for fn in filelist:
  d = torch.load(fn)
  for i in range(len(d)):
      dist = d[i].distances_emb.numpy()
      ey = d[i].edges_y.numpy()
      s = d[i].score.numpy()
      ne = ((dist<distcut) & (s>scorecut)).sum()
      ne_truth = ((dist<distcut) & (s>scorecut) & (ey==True)).sum()
      ne_false = ((dist<distcut) & (s>scorecut) & (ey==False)).sum()
      distance.append(dist)
      score.append(s)
      edge_y.append(ey)
      nedges.append(ne)
      nedges_matched.append(ne_truth)
      nedges_nomatched.append(ne_false)
  del d

distance = np.concatenate(distance)
score = np.concatenate(score)
edge_y = np.concatenate(edge_y)
nedges = np.array(nedges)
nedges_matched = np.array(nedges_matched)
nedges_nomatched = np.array(nedges_nomatched)

plt_dist(distance,edge_y)
plt_score(score,edge_y)
plt_Nedges(nedges,nedges_matched,nedges_nomatched)
