import itertools
import sys
import subprocess
import os


architectures=["ginconv", "leconv", "gcn", "sageconv", "attention"]
node_features=["connection"]
mlp_hid = [16, 32, 64, 128]
threshold=[0.5, 1, 1.5, 2, 2.5, 3]
num_layers=[1, 2, 3]
dropout=[0.1, 0.05]
n_hid = [8, 16, 32]
lr_gnn=[0.01, 0.001, 0.0001]
combinations = list(itertools.product(architectures, node_features, mlp_hid, threshold, num_layers, dropout, n_hid, lr_gnn))

index_string = sys.argv[1]
pbs_index = int(index_string)
combination_index = pbs_index-1

selected_combination = combinations[combination_index]

result=subprocess.run(
    ["python", "training_sMRI.py",
     '--arch', str(selected_combination[0]), 
     '--node_feature', str(selected_combination[1]),
     '--MLP_hid', str(selected_combination[2]),
     '--threshold', str(selected_combination[3]),
     '--num_layers', str(selected_combination[4]), 
     '--dropout', str(selected_combination[5]), 
     '--nhid', str(selected_combination[6]),                 
     '--lr_gnn', str(selected_combination[7]),
     '--PBS_ID', str(pbs_index)])
