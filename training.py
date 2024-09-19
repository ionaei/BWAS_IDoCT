import os
import pandas as pd
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from time import time
from pathlib import Path
from typing import NamedTuple, Optional, Any
from sklearn.metrics import roc_auc_score, f1_score
from torch.nn.functional import normalize
from tqdm import tqdm

from dataset import GraphDataset
from torch.utils.data import DataLoader
#model
import model



#make the graphs
import generate_graphs_func_val

#train function
import train

#test function
import test

import parser

#get data

architectures=parser.args.arch
node_features=parser.args.node_feature
threshold=parser.args.threshold
num_layers=parser.args.num_layers
dropout=parser.args.dropout
n_hid = parser.args.nhid
lr_gnn=parser.args.lr_gnn

config = {
    "architecture": architectures,
    "node_feature": node_features,
    "threshold": threshold,
    "num_layers": num_layers,
    "dropout": dropout,
    "learning_rate": lr_gnn,
    "n_hid": n_hid
}

os.environ["WANDB__SERVICE_WAIT"] = "300"


pbs_ID = parser.args.PBS_ID

#folder location to save checkpoint
save_dir = "location/checkpoint/folder/"

checkpoint_filepath = os.path.join(save_dir,  f"cognitive_score_{pbs_ID}.pth")


cog_directory = "location of cognitive data in csv form"
image_directory = 'location of rs-fMRI data matrices'

#under outcome, options are "AS" or "DT" depending which investigated trait is of interest
graphs = generate_graphs_func_val.generate_graphs(directory_graphs=image_directory, dim=21, node_type=parser.args.node_feature, path_to_csv=cog_directory, outcome='AS', threshold_edge=parser.args.threshold)

dataset = GraphDataset(graphs=graphs, split_sizes=(0.7, 0.2, 0.1), seed=912, device='cpu')

train_loader = dataset.get_train_loader(batch_size=200)
val_loader = dataset.get_val_loader()
test_loader = dataset.get_test_loader()

mod = model.Model(nfeat=graphs[0].x.shape[1], nhid=parser.args.nhid, nclass=1, dropout=parser.args.dropout, num_layers=parser.args.num_layers, gnn_arch=parser.args.arch)

criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(mod.parameters(), lr=parser.args.lr_gnn)


# Save checkpoint function
def save_checkpoint(model, optimizer, epoch, val_r2, filepath="checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_r2': val_r2
    }
    torch.save(checkpoint, filepath)


# Training and testing

epochs = 5000
train_losses = []
train_r2 = []
val_losses = []
val_r2_tot = []

best_val_r2 = -float('inf')  

for i in range(epochs):
    print(f"Epoch {i+1}\n-------------------------------")
    epoch_losses, epoch_r2 = train.train_loop(dataloader=train_loader, model=mod, criterion=criterion, optimizer=optimizer, epoch=i)
    print("Validation")
    _, _, val_mse, val_r2 = test.eval_loop(dataloader=val_loader, model=mod)
    
    """wandb.log({'train loss mse': epoch_losses})
    wandb.log({'train r2': epoch_r2})
    wandb.log({'val loss mse': val_mse})
    wandb.log({'val R2': val_r2})"""

    train_losses.append(epoch_losses)
    train_r2.append(epoch_r2)
    val_losses.append(val_mse)
    val_r2_tot.append(val_r2)
    
    # Check if current validation R2 is the best
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        
        
        save_checkpoint(mod, optimizer, i, best_val_r2, filepath=checkpoint_filepath)
        print(f"New best model saved at epoch {i+1} with validation R2: {val_r2}")

print("Test")

#loading of best model from the checkpoint file path
checkpoint = torch.load(checkpoint_filepath)
mod.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
best_epoch = checkpoint['epoch']
best_val_r2 = checkpoint['val_r2']

print(f"Best model from epoch {best_epoch+1} with validation R2: {best_val_r2}")

_, _, test_mse, test_r2 = test.eval_loop(dataloader=test_loader, model=mod)

print("Test R2:", test_r2)
print("Test MSE:", test_mse)
print("Best epoch:", best_epoch)
print("Best val R2:", best_val_r2)


