import torch
import torch_geometric as PyG


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        # 第1层图卷积（局部特征提取）
        self.gnn1 = PyG.nn.SAGEConv(input_dim, 128)
        # 第2层图卷积
        self.gnn2 = PyG.nn.SAGEConv(128, 256)
        # 图Transformer（全局交互）
        self.transformer1 = PyG.nn.TransformerConv(256, 32, heads=8)
        self.transformer2 = PyG.nn.TransformerConv(256, 32, heads=8)
        # 特征投影层（输出固定维度嵌入）
        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, hidden_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 图卷积提取局部特征
        x = self.gnn1(x, edge_index)
        x = torch.relu(x)
        x = self.gnn2(x, edge_index)
        x = torch.relu(x)

        # Transformer捕获全局交互
        x = self.transformer1(x, edge_index)
        x = torch.relu(x)
        x = self.transformer2(x, edge_index)
        x = torch.relu(x)

        # 投影到目标维度
        x = self.proj(x)
        # 全局池化（将整个图的节点特征聚合成一个向量）
        return PyG.nn.global_max_pool(x, data.batch)


class Network(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, get_embeddings):
        super(Network, self).__init__()
        self.gnn = GNN(input_dim, hidden_dim)

        # 若为“提取嵌入”模式，直接返回GNN输出；否则走分类头
        if get_embeddings:
            self.head = torch.nn.Identity()
        else:
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 1),
                torch.nn.Sigmoid()
            )

    def forward(self, data):
        gnn_emb = self.gnn(data)
        return self.head(gnn_emb)


def create_gnn_model(config, get_embeddings=False):
    model = Network(
        input_dim=config['network']['GNN']['input_dim'],
        hidden_dim=config['network']['GNN']['hidden_dim'],
        get_embeddings=get_embeddings
    ).to(config['device'])
    return model


def cri_opt_sch(config, model):
    # 二分类损失（适配预训练的属性预测任务）
    criterion = torch.nn.BCELoss()
    # AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])

    # 学习率调度器
    if config['sch']['name'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['optim']['lr'],
            epochs=config['epochs'],
            steps_per_epoch=config['sch']['steps']
        )
    elif config['sch']['name'] == 'lronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # 损失越小越好，故用min模式
            factor=config['sch']['factor'],
            patience=config['sch']['patience']
        )
    else:
        # 默认调度器（可根据需求扩展）
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    return criterion, optimizer, scheduler