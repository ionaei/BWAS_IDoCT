from sklearn.metrics import r2_score

import torch.nn.functional as F
import torch

def eval_loop(dataloader, model):
    model.eval()
    preds = []
    actual = []
    
    r2_scores =[]

    
    for data in dataloader:
        
        x = data[0].x.float()
        edge_index = data[0].edge_index.long()
        edge_attr = data[0].edge_weight.float()
        batch = data[0].batch
        
        smri_data = data[1].float()
        
        out = model(x, edge_index, edge_attr, batch=batch, smri_data=smri_data)
        pred = out.squeeze(dim=1)
        preds.append(pred)
        actual.append(data[0].y)
        
    
    preds = torch.cat(preds, dim = 0)
    
    actual = torch.cat(actual)
    
    mse = F.mse_loss(preds, actual)
    
    
    print(f"MSE: {mse:.6f}")
    
    #different calculation of r2
    preds_np = preds.detach().numpy()
    
    actual_np = actual.numpy()   
    
    r2 = r2_score(actual_np, preds_np)
    
    
    print(f"R2: {r2: .6f}")
    
    return preds_np, actual_np, mse, r2
    
