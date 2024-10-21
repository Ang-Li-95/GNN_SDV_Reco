import os
import logging

import torch
from torch.utils.data import random_split
from torch import nn
import scipy as sp
import numpy as np

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

import faiss
import faiss.contrib.torch_utils

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def load_dataset(input_dir,max_events_per_source=None):
  assert os.path.exists(input_dir), "Dir {} does not exist!".format(input_dir)
  total_events = []
  if os.path.isdir(input_dir):
    for fn in os.listdir(input_dir):
    #for root, dirs, fns in os.walk(input_dir):
    #  for fn in fns:
        if not fn.endswith('.pt'):
          continue
        full_fn = os.path.join(input_dir,fn)
        events = torch.load(full_fn, map_location=torch.device("cpu"))
        if type(events)==list or type(events)==torch.utils.data.dataset.Subset:
          if max_events_per_source is not None:
            events_take = min(max_events_per_source,len(events))
            if not type(events)==list:
              events = list(events)
            events = events[:events_take]
          total_events += events
        elif type(events)==Data:
          total_events.append(events)
        else:
          raise TypeError("Unrecognized data type.")
  elif os.path.isfile(input_dir):
    full_fn = input_dir
    if not full_fn.endswith('.pt'):
      raise TypeError("Unrecognized data file {}.".format(full_fn))
    events = torch.load(full_fn, map_location=torch.device("cpu"))
    if type(events)==list or type(events)==torch.utils.data.dataset.Subset:
      if max_events_per_source is not None:
        events_take = min(max_events_per_source,len(events))
        events = events[:events_take]
      total_events += events
    elif type(events)==Data:
      total_events.append(events)
    else:
      raise TypeError("Unrecognized data type.")
  return total_events

def split_dataset(input_dir,split,seed=None,max_events_per_source=None):
  if seed:
    torch.manual_seed(seed)
  dataset = load_dataset(input_dir,max_events_per_source)
  train_events, val_events, test_events = random_split(dataset, split)
  return train_events, val_events, test_events

def mlp(
    node_sizes,
    hidden_activation,
    output_activation,
    layer_norm
    ):
  """
  This function constructs multilayer perceptron.
  node_sizes: a list that includes the number of nodes in each layer
  hidden_activation: activation function used in hidden layers
  output_activation: activation function used in output layer
  layer_norm: bool value that represents whether to perform a LayerNorm after each layer
  """
  hidden_activation = getattr(nn,hidden_activation)
  if output_activation is not None:
    output_activation = getattr(nn,output_activation)
  layers = []
  for i in range(0,len(node_sizes)-2):
    layers.append(nn.Linear(node_sizes[i],node_sizes[i+1]))
    if layer_norm:
      layers.append(nn.LayerNorm(node_sizes[i+1]))
      layers.append(hidden_activation())
  layers.append(nn.Linear(node_sizes[-2],node_sizes[-1]))
  if output_activation is not None:
    if layer_norm:
      layers.append(nn.LayerNorm(node_sizes[-1]))
    layers.append(output_activation())
  return nn.Sequential(*layers)

def build_edges(query_nodes,all_nodes,indices=None, r_max=1.0, k_max=10, return_indices=False):

    # Calculate the distance between query node and other nodes in the latend space
    # d2 represents the squared distance and idx represents the index of nodes that are close with the query node
    if device == "cuda":
        res = faiss.StandardGpuResources()
        d2, idx = faiss.knn_gpu(res=res, xq=query_nodes, xb=all_nodes, k=k_max)
    elif device == "cpu":
        index = faiss.IndexFlatL2(all_nodes.shape[1])
        index.add(all_nodes)
        d2, idx = index.search(query_nodes, k_max) 
    #index = faiss.IndexFlatL2(all_nodes.shape[1])
    #index.add(all_nodes)
    #d2, idx = index.search(query_nodes, k_max) 

    # Now constrct a tensor that represent the indices of query nodes
    query_idx = torch.Tensor.repeat(
        torch.arange(idx.shape[0], device=device), (idx.shape[1], 1), 1
    ).T.int()

    edge_list = torch.stack([query_idx[d2<= r_max**2], idx[d2 <= r_max**2]])

    # Transfer query_idx to their corresponding indices in all_nodes
    if indices is not None:
        edge_list[0] = indices[edge_list[0]]

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]

    if return_indices:
        return edge_list, d2, idx, query_idx
    else:
        return edge_list

def graph_intersection(
    pred_graph, truth_graph, using_weights=False, weights_bidir=None
):
    if pred_graph.shape[1]==0 and truth_graph.shape[1]==0:
      array_size = 1
    elif pred_graph.shape[1]==0:
      array_size = truth_graph.max().item() + 1
    elif truth_graph.shape[1]==0:
      array_size = pred_graph.max().item() + 1
    else:
      array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)
    del e_1
    del e_2

    if using_weights:
        weights_list = weights_bidir.cpu().numpy()
        weights_sparse = sp.sparse.coo_matrix(
            (weights_list, l2), shape=(array_size, array_size)
        ).tocsr()
        del weights_list
        del l2
        new_weights = weights_sparse[e_intersection.astype("bool")]
        del weights_sparse
        new_weights = torch.from_numpy(np.array(new_weights)[0])

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long().to(device)
    y = torch.from_numpy(e_intersection.data > 0).to(device)
    del e_intersection

    if using_weights:
        return new_pred_graph, y, new_weights
    else:
        return new_pred_graph, y
