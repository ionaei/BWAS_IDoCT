from generate_node_features_func import generate_node_features
from extract_labels_func import extract_labels
import numpy as np
import torch
import os
import pandas as pd
import glob
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse



def generate_graphs(directory_graphs, dim, node_type, path_to_csv, outcome, threshold_edge):
    lisfiles = glob.glob(f'{directory_graphs}/*txt')
    
    labels_dict = extract_labels(directory_graphs, path_to_csv, outcome)
    print("Number of keys", len(labels_dict.keys()))
    
    dataset = []
    for file in lisfiles:
        user_id = os.path.basename(file).split("_")[0][1:]
        
        #Extract tp2
        if int(os.path.basename(file).split("_")[2]) == 3:
            continue
        user_id = int(user_id)
        if user_id in labels_dict.keys():
            
            vector_from_file = np.loadtxt(file, unpack=False)
            square_net_from_file = np.zeros((dim, dim))
            square_net_from_file[np.triu(np.ones(dim), k=1).astype(bool)] = vector_from_file
            adj_matrix = square_net_from_file + square_net_from_file.T
            
            label = torch.tensor(labels_dict[user_id])
            node_feature = generate_node_features(node_type = node_type, corr_matrix = square_net_from_file)
            
            adj_matrix[abs(adj_matrix)< threshold_edge] = 0
            dense_tensor = dense_to_sparse(torch.tensor(adj_matrix))
            
            edge_index = dense_tensor[0]
            edge_weight = dense_tensor[1]
            edge_weight_pos = edge_weight - edge_weight.min()
            edge_weight_norm = edge_weight_pos / edge_weight_pos.max()
            
            data = Data(x=node_feature, edge_index=edge_index, edge_weight = edge_weight_norm, y = label, 
                       adj_matrix = adj_matrix, eid = user_id, num_node_features = node_feature.shape[0])
            dataset.append(data)
    print("Number of graphs", len(dataset))
    return dataset
