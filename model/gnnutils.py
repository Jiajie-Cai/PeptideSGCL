import torch

def train_gnn(model, train_loader, optimizer, criterion, scheduler, device):
    """GNN训练单轮"""
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(), data.y)  # out是[batch_size,1]，data.y是[batch_size]
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs  # 按图数量加权
    return total_loss / len(train_loader.dataset)  # 平均损失


def validate_gnn(model, val_loader, criterion, device):
    """GNN验证单轮"""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out.squeeze(), data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(val_loader.dataset)