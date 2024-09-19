import pandas as pd
import os
import glob

def extract_labels(directory_graphs, path_to_csv, outcome):
    df = pd.read_csv(path_to_csv)
    df.rename(columns = {"AS_6364_2_0": "AS", "DT_6364_2_0": "DT"}, inplace = True)
    df = df.dropna(subset = ["AS", "DT"])
    
    labels_dict = {}
    print(directory_graphs)
    user_ids = [os.path.basename(file).split("_")[0][1:] for file in glob.glob(f'{directory_graphs}/*txt')]
    
    
    for user in user_ids:
        if int(user) in df.eid.tolist():
            labels_dict[int(user)] = df[df.eid == int(user)][outcome].item()
        
    return labels_dict
