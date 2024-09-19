"""This code is for the elastic net regression for AS prediction. 
To change to DT change the outcome variable from AS to DT in the generate_graphs_func.generate_graphs"""

import os
import pandas as pd
import glob
import numpy as np

#make the graphs
import generate_graphs_func
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from itertools import product


cog_directory = "location of cognitive csv file"
image_directory = 'location of rs-fMRI matrices'

"""set node_type and threshold as determined best value from rs-fMRI model. 
set outcome as AS or DT"""

graphs = generate_graphs_func.generate_graphs(directory_graphs = image_directory, dim = 21, node_type = 'node type', path_to_csv = cog_directory, outcome = 'AS', threshold_edge = 0.5)

node_features = []
edge_features = []
AS = [] #change accordingly for DT
for graph in graphs:
    np_node_feature = graph.x.numpy()
    node_features.append(np_node_feature)
    edge_features.append(graph.adj_matrix)
    AS.append(graph.y.item())
    
node_features_flattened = np.array([feature.flatten() for feature in node_features])    
edge_features_flattened = np.array([feature.flatten() for feature in edge_features])

nodes_df = pd.DataFrame(node_features_flattened)
edge_df = pd.DataFrame(edge_features_flattened)

total_df = pd.concat([nodes_df, edge_df], axis=1)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(total_df)
scaled_df = pd.DataFrame(scaled_data, columns=total_df.columns)

pca = PCA(n_components=scaled_df.shape[1])
pca.fit(scaled_df)

eigenvalues = pca.explained_variance_
kaiser_criterion_indices = np.where(eigenvalues > 1)[0]
n_components_kaiser = len(kaiser_criterion_indices)

print(f"Number of components selected by Kaiser criterion: {n_components_kaiser}")

pca_final = PCA(n_components=n_components_kaiser) #based on kaiser criterion
scaled_df_pca = pca_final.fit_transform(scaled_df)

df_AS = pd.DataFrame(scaled_df_pca)
df_AS['AS'] = AS


#ELASTIC NET REGRESSION FOR AS 

y_AS = df_AS['AS']
features_AS = df_AS.drop('AS', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features_AS, y_AS, test_size=0.2, random_state=42)

# Further split training data into training and validation sets
X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

elastic_net = ElasticNet()

# Define parameter grid
param_grid = {
    'alpha': np.logspace(-3, 2, 10),  
    'l1_ratio': np.linspace(0.1, 1, 10)
}

best_score = -float('inf')
best_params = None

for alpha, l1_ratio in product(param_grid['alpha'], param_grid['l1_ratio']):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train_main, y_train_main) 
    y_pred_val = model.predict(X_val)      
    score = r2_score(y_val, y_pred_val)
    
    if score > best_score:
        best_score = score
        best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}

print(f'\nBest parameters: {best_params}')

# Use the best parameters to train the final model on the full training set (X_train)
best_alpha = best_params['alpha']
best_l1_ratio = best_params['l1_ratio']

final_model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
final_model.fit(X_train, y_train)  

y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'\nTrain R² score: {train_r2}')
print(f'Test R² score: {test_r2}')



    
