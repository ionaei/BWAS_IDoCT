from sklearn.metrics import r2_score

def train_loop(dataloader, model, criterion, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    losses = []
    r2_scores = []
    
    for data in dataloader: 
        optimizer.zero_grad()
        
        x = data[0].x.float()
        edge_index = data[0].edge_index.long()
        edge_attr = data[0].edge_weight.float()
        batch = data[0].batch
        
        smri_data = data[1].float()  
        
        # Forward pass
        out = model(x, edge_index, edge_attr, batch=batch, smri_data=smri_data)
        
        # Compute loss
        loss = criterion(out.squeeze(), data[0].y.float())
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step() 
        
        losses.append(loss.item())  # Append the loss value
        
        # Calculate R2 score
        preds_np = out.squeeze().detach().numpy()
        actual_np = data[0].y.numpy()
        
        r2 = r2_score(actual_np, preds_np)
        r2_scores.append(r2)
    
    # Compute average loss and R2 score for the epoch
    avg_loss = sum(losses) / len(losses)
    avg_r2 = sum(r2_scores) / len(r2_scores)
    
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}, R2: {avg_r2:.6f}")
    
    return avg_loss, avg_r2
