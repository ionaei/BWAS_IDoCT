import numpy as np
import torch

def generate_node_features(node_type, corr_matrix):
    if node_type == "identity":
        # Generate a unique one hot vector per node
        node_feature = torch.diag(torch.ones(corr_matrix.shape[0]))
    if node_type == "connection":
        # Use the weighted connection with the other ICA
        adj_matrix = corr_matrix + corr_matrix.T
        node_feature = adj_matrix
    if node_type == "eigen":
        adj_matrix = corr_matrix + corr_matrix.T
        w, v = np.linalg.eig(adj_matrix)
        node_feature = v.transpose()    
    if node_type == "degree":
        adj_matrix = corr_matrix + corr_matrix.T
        node_feature = torch.tensor(adj_matrix).sum(1, keepdim=True)
    node_feature = torch.tensor(node_feature).float()
    return node_feature
