from sklearn.metrics import r2_score

import torch.nn.functional as F
import torch

def eval_loop(dataloader, model):
    model.eval()
    preds = []
    actual = []
    
    for data in dataloader:
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        pred = out.squeeze(dim=1)
        preds.append(pred)
        actual.append(data.y)
    
    preds = torch.cat(preds, dim = 0)
    actual = torch.cat(actual)
    mse = F.mse_loss(preds, actual)
      
    print(f"MSE: {mse:.6f}")
    
    preds_np = preds.detach().numpy()
    
    actual_np = actual.numpy()   
    
    r2 = r2_score(actual_np, preds_np)
    print(f"R2: {r2: .6f}")
    
    return preds_np, actual_np, mse, r2
    
