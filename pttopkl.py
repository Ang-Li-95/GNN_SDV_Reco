import torch
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', help='Directory that includes the data to be processed [NANOAOD]')
parser.add_argument('--output_file', help='Directory of output')
args = parser.parse_args()

d = torch.load(args.input_file)
distance = []
score = []
edge_y = []
for i in range(len(d)):
    dist = d[i].distances_emb.numpy()
    ey = d[i].edges_y.numpy()
    s = d[i].score.numpy()
    distance.append(dist)
    score.append(s)
    edge_y.append(ey)

distance = np.concatenate(distance)
score = np.concatenate(score)
edge_y = np.concatenate(edge_y)

output = {
    'distance':distance,
    'score':score,
    'edge_y':edge_y,
         }

with open(args.output_file,'wb') as f:
    pickle.dump(output,f)
