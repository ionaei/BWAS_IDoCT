from sklearn.metrics import r2_score

def train_loop(dataloader, model, criterion, optimizer, epoch):
    model.train()
    losses = []
    r2_scores = []
    
    for data in dataloader: 
        optimizer.zero_grad()
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_weight.float()
        batch = data.batch.float()
                
        #prediction and loss
        out = model(x, edge_index, edge_attr, batch)
        
        loss = criterion(out.squeeze(), data.y)
        
        #backprop
        loss.backward()
        optimizer.step() 
        losses.append(loss)  
       
        preds_np = out.squeeze().detach().numpy()
        actual_np = data.y.numpy()
        
        r2 = r2_score(actual_np, preds_np)
        r2_scores.append(r2)

    
    avg_loss = sum(losses) / len(losses)  # Average mse loss
    avg_r2 = sum(r2_scores) / len(r2_scores) #average r2
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, R2: {avg_r2: .6f}")

    
    return avg_loss , avg_r2
    
