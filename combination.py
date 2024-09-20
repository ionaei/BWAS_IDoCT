#combination for hyperparameter tuning

import itertools
import sys
import subprocess
import os

architectures=["ginconv", "leconv", "gcn", "sageconv", "attention"]
node_features=["connection"]
threshold=[0.5, 1, 1.5, 2, 2.5, 3]
num_layers=[1, 2, 3]
dropout=[0.1, 0.05]
n_hid = [8, 16, 32]
lr_gnn=[0.01, 0.001, 0.0001]

combinations = list(itertools.product(architectures, node_features, threshold, num_layers, dropout, n_hid, lr_gnn))

index_string = sys.argv[1]
pbs_index = int(index_string)
combination_index = pbs_index-1

selected_combination = combinations[combination_index]


result=subprocess.run(
    ["python", "training.py",
     '--arch', selected_combination[0], 
     '--node_feature', selected_combination[1], 
     '--threshold', str(selected_combination[2]),
     '--num_layers', str(selected_combination[3]), 
     '--dropout', str(selected_combination[4]), 
     '--nhid', str(selected_combination[5]),                 
     '--lr_gnn', str(selected_combination[6]),
     '--PBS_ID', str(pbs_index)])
