# -*- coding="utf-8" -*-

import json
import glob
import numpy as np
from utils import *
from tqdm import  tqdm
import torch

def load_data(args):
    print("\nEnumerating feature and target values.\n")
    ending="*.json"

    train_graph_paths=glob.glob(args.train_graph_folder+ending)
    test_graph_paths=glob.glob(args.test_graph_folder+ending)

    graph_paths=train_graph_paths+test_graph_paths

    targets=set()
    features=set()

    for path in tqdm(graph_paths):
        data=json.load(open(path))
        targets=targets.union(set([data["target"]]))
        features=features.union(set(data["labels"]))

    target_map=create_numeric_mapping(targets)
    feature_map=create_numeric_mapping(features)

    # number_of_features=len(feature_map)
    # number_of_targets=len(target_map)


    return train_graph_paths,test_graph_paths,feature_map,target_map

def create_target(data,target_map):


    return torch.FloatTensor([0.0 if i != data["target"] else 1.0 for i in range(len(target_map))])

def create_edges(data):
    N=len(data["labels"].keys())
    M=len(data["edges"])*2
    temp_edges = [[edge[0], edge[1]] for edge in data["edges"]] + [[edge[1], edge[0]] for edge in data["edges"]]
    temp_edges=list(zip(*temp_edges))
    edges=np.zeros((N,N),dtype=np.float32)
    for i  in range(M):
        row=temp_edges[0][i]
        col=temp_edges[1][i]
        edges[row,col]=1
    return torch.FloatTensor(edges)

def create_features(data,feature_map):
    features=np.zeros((len(data["labels"]),len(feature_map)),dtype=np.float32)
    node_indices = [node for node in range(len(data["labels"]))]
    feature_indices = [feature_map[label] for label in data["labels"].values()]
    features[node_indices, feature_indices] = 1.0
    features = torch.FloatTensor(features)
    return features

def create_input_data(path,feature_map,target_map):

    data=json.load(open(path))
    target=create_target(data,target_map)
    edges=create_edges(data)
    features=create_features(data,feature_map)
    to_pass_forward=dict()
    to_pass_forward["target"]=target
    to_pass_forward["edges"]=edges
    to_pass_forward["features"]=features
    return to_pass_forward