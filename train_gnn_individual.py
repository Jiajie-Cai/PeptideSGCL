import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
from typing import Dict, Any
from torch_geometric.loader import DataLoader


from data.gnn_dataloader import load_data
from model.network_gnn import create_gnn_model, cri_opt_sch
from model.gnnutils import train_gnn, validate_gnn


def get_device() -> torch.device:
    """获取可用设备（优先GPU）"""
    if torch.cuda.is_available():
        # 支持多GPU选择（从配置或默认）
        gpu_id = config.get('gpu_id', 3)
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def train_gnn_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Dict[str, Any],
    device: torch.device
) -> str:
    """训练GNN模型（已移除早停机制）"""
    print(f'{"=" * 30}{"GNN PRETRAINING":^20}{"=" * 30}')

    best_val_loss = float('inf')  # 仅用于保存最优模型，不触发早停

    # 模型移动到设备
    model.to(device)

    for epoch in range(config['epochs']):
        # 训练阶段
        train_loss = train_gnn(model, train_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{config["epochs"]} - Train Loss: {train_loss:.4f} \tLR: {curr_lr}')

        # 验证阶段
        val_loss = validate_gnn(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{config["epochs"]} - Validation Loss: {val_loss:.4f}\n')

        # 学习率调度
        if config['sch']['name'] == 'lronplateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # WandB日志
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': curr_lr
            })

        # 保存最优模型（仅根据验证损失，不触发早停）
        save_dir = f'./checkpoints/individual_pretrained/balanced_data/GNN/{config["task"]}'
        os.makedirs(save_dir, exist_ok=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'gnn_state_dict': model.gnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': curr_lr
            }, os.path.join(save_dir, 'model.pt'))
            print('GNN Model Saved (最优模型更新)\n')

    return 'GNN Pretraining completed (已完成所有epochs)'


if __name__ == "__main__":
    # 任务配置
    task_name = "hemo"
    config_path = f'./checkpoints/individual_pretrained/balanced_data/GNN/{task_name}/config.yaml'

    # 加载配置
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config: Dict[str, Any] = yaml.load(f, Loader=yaml.FullLoader)

    # 设备配置
    device = get_device()
    config['device'] = device
    print(f'Device: {device}\n')

    # 加载数据
    train_data_loader, val_data_loader = load_data(config)
    config['sch']['steps'] = len(train_data_loader)

    # 创建模型
    model = create_gnn_model(config, get_embeddings=False)

    # 损失、优化器、调度器
    criterion, optimizer, scheduler = cri_opt_sch(config, model)

    # 保存配置与代码
    save_dir = f'./checkpoints/individual_pretrained/balanced_data/GNN/{config["task"]}'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy('./model/network_gnn.py', os.path.join(save_dir, 'network.py'))
    with open(os.path.join(save_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)

    # WandB初始化
    if not config['debug']:
        run_name = f'gnn_pretrain_{config["task"]}_{datetime.now().strftime("%m%d_%H%M")}'
        wandb.init(project='GNN_Pretrain_Peptide', name=run_name, config=config)

    # 执行训练
    train_gnn_model(model, train_data_loader, val_data_loader, criterion, optimizer, scheduler, config, device)
    if not config['debug']:
        wandb.finish()
