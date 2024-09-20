import pandas as pd
import generate_graphs_func
import torch
import torch.nn as nn

from torch_geometric.data import Data

import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from time import time
from pathlib import Path
from typing import NamedTuple, Optional, Any
from tqdm import tqdm

from dataset_split import GraphDataset
from torch.utils.data import DataLoader
#model
import model_fMRI_sMRI

#train function
import train_sMRI

#test function
import test_sMRI

from sklearn.preprocessing import MinMaxScaler

import parser_sMRI

#get data

architectures=parser_sMRI.args.arch
node_features=parser_sMRI.args.node_feature
threshold=parser_sMRI.args.threshold
num_layers=parser_sMRI.args.num_layers
dropout=parser_sMRI.args.dropout
n_hid = parser_sMRI.args.nhid
lr_gnn=parser_sMRI.args.lr_gnn
mlp_hid = parser_sMRI.args.MLP_hid


config = {
    "architecture": architectures,
    "node_feature": node_features,
    "threshold": threshold,
    "num_layers": num_layers,
    "dropout": dropout,
    "learning_rate": lr_gnn,
    "n_hid": n_hid,
    "mlp_hid": mlp_hid
}


pbs_ID = parser_sMRI.args.PBS_ID


save_dir = "Location of directory to save checkpoints to"
checkpoint_filepath = os.path.join(save_dir,  f"best_sMRI_{pbs_ID}.pth")

df = pd.read_excel("sMRI dataframe location")

cog_directory = "cognitive data csv file location"
image_directory = 'rs-fMRI matrices folder location'

#set outcome as AS or DT
graphs = generate_graphs_func.generate_graphs(directory_graphs = image_directory, dim = 21, node_type = node_features, path_to_csv = cog_directory, outcome = 'AS', threshold_edge = threshold)

ids = []
for graph in graphs:
    ids.append(graph.eid)
    
matched_df = df[df['eid'].isin(ids)]

#remove the 220 graphs with missing data

nan_indices = matched_df[matched_df['Number_of_HolesBeforeFixing_left_hemisphere_adj'].isna()].index
nan_indicies = matched_df.loc[nan_indices,'eid']

nan_eid = nan_indicies.tolist()

for i in range(len(nan_eid)):
    for graph in graphs:
        if graph.eid == nan_eid[i]:
            graphs.remove(graph)

print(len(graphs))

matched_nonan_df = matched_df.dropna()

ids_new = []
for graph in graphs:
    ids_new.append(graph.eid)
    
matched_nonan_df['eid'] = pd.Categorical(matched_nonan_df['eid'], categories=ids_new, ordered=True)
matched_nonan_df=matched_nonan_df.sort_values('eid').reset_index(drop=True)

fa_columns = [str(col) for col in matched_nonan_df.columns if str(col).startswith('FA_')]
mean_columns = [str(col) for col in matched_nonan_df.columns if str(col).startswith('Mean_thickness')]
grey_columns = [str(col) for col in matched_nonan_df.columns if str(col).startswith('Volume_of_grey_matter')]
intensity_columns = [str(col) for col in matched_nonan_df.columns if str(col).startswith('Mean_intensity')]

print(f"There are {len(fa_columns)} FA, {len(mean_columns)} thickness, {len(grey_columns)} grey and {len(intensity_columns)} intensity")

FA_df = matched_nonan_df[fa_columns]
thickness_df = matched_nonan_df[mean_columns]
grey_df = matched_nonan_df[grey_columns]
intensity_df = matched_nonan_df[intensity_columns]

eid=matched_nonan_df['eid']

total_features = pd.concat([FA_df, thickness_df, grey_df, intensity_df], axis=1)
total_features.insert(0, 'eid', eid)

features = total_features.drop('eid', axis = 1)
features_mlp = torch.tensor(features.values, dtype=torch.float)

train_index = int(features_mlp.shape[0]*0.7)
test_index = int(features_mlp.shape[0]*0.2)
val_index = int(features_mlp.shape[0]-train_index-test_index)

train_df = features.iloc[:train_index]
test_df = features.iloc[train_index: test_index+train_index]
val_df = features.iloc[test_index+train_index:]

scaler = MinMaxScaler()

train_df_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
test_df_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
val_df_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns)

scaled_smri_features = pd.concat([train_df_scaled, test_df_scaled, val_df_scaled], ignore_index=True)
scaled_features_smri = torch.tensor(scaled_smri_features.values, dtype=torch.float)

#graph dataloader

all_indices = torch.arange(start=0, end=features.shape[0])
train_indices = all_indices[:train_index]
test_indices = all_indices[train_index:train_index + test_index]
val_indicies = all_indices[train_index + test_index:]


dataset = GraphDataset(graphs=graphs, smri_data=scaled_features_smri, split_indices=(train_indices, val_indicies, test_indices), seed=912, device='cpu')
train_loader = dataset.get_train_loader(batch_size=200)
val_loader = dataset.get_val_loader()
test_loader = dataset.get_test_loader()

#model

mod = model_fMRI_sMRI.Model(nfeat= graphs[0].x.shape[1], nhid=n_hid, nclass=1, dropout=dropout, num_layers=num_layers, gnn_arch=architectures, smri_input_size=features.shape[1], smri_hidden_size=mlp_hid)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(mod.parameters(), lr=lr_gnn)


def save_checkpoint(model, optimizer, epoch, val_r2, filepath="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_r2': val_r2
    }
    torch.save(checkpoint, filepath)
    
    

epochs = 4000
train_losses = []
train_r2 = []
val_losses = []
val_r2_tot = []

best_val_r2 = -float('inf')  

for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    epoch_losses, epoch_r2 = train_sMRI.train_loop(dataloader=train_loader, model=mod, criterion=criterion, optimizer=optimizer, epoch=i)
    print("Validation")
    _, _, val_mse, val_r2 = test_sMRI.eval_loop(dataloader=val_loader, model=mod)
    


    train_losses.append(epoch_losses)
    train_r2.append(epoch_r2)
    val_losses.append(val_mse)
    val_r2_tot.append(val_r2)
    
    # Check if current validation R2 is the best
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        
        
        save_checkpoint(mod, optimizer, i, best_val_r2, filepath=checkpoint_filepath)
        print(f"New best model saved at epoch {i+1} with validation R2: {val_r2}")
        
        checkpoint = torch.load(checkpoint_filepath)
        mod.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_epoch = checkpoint['epoch']
        best_val_r2 = checkpoint['val_r2']

        print(f"Evaluating best model from epoch {best_epoch+1} on test set")
        
        _, _, test_mse, test_r2 = test_sMRI.eval_loop(dataloader=test_loader, model=mod)
        
        print(f"Test mse: {test_mse}, Test R2: {test_r2}")


