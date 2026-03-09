import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as PyG
from transformers import BertModel, BertConfig, logging
logging.set_verbosity_error()
import torch_geometric.nn as PyG_nn

class ParallelGraphSCAttention(nn.Module):
    def __init__(self, channels: int, groups: int = None, hidden: int = 8,
                 use_graphnorm: bool = True, reduction: int = 4,
                 attn_dropout: float = 0.0, sparse_ratio: float = 0.0, temp: float = 1.0):
        super().__init__()
        if groups is None:
            groups = min(16, max(4, channels // 16))

        assert channels % (2 * groups) == 0, f"channels ({channels}) must be divisible by 2*groups ({2 * groups})"
        self.channels = channels
        self.groups = groups
        self.group_c = channels // groups
        self.half_group_c = self.group_c // 2
        assert self.half_group_c >= 4, f"half_group_c ({self.half_group_c}) too small"

        gh = self.groups * self.half_group_c
        hidden_dim = max(gh // reduction, 8)

        self.cweight = nn.Parameter(torch.zeros(1, self.half_group_c))
        self.cbias = nn.Parameter(torch.ones(1, self.half_group_c))
        self.sweight = nn.Parameter(torch.zeros(1, self.half_group_c))
        self.sbias = nn.Parameter(torch.ones(1, self.half_group_c))


        self.channel_mlp = nn.Sequential(
            nn.Linear(gh * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, gh)
        )

        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=gh)

        self.space_mlp = nn.Sequential(
            nn.Linear(self.half_group_c * 3, max(self.half_group_c // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(self.half_group_c // reduction, 8), self.half_group_c)
        )

        self.register_parameter("init_scale", nn.Parameter(torch.tensor(0.0)))
        self.sigmoid = nn.Sigmoid()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.sparse_ratio = float(sparse_ratio)
        self.temp = float(temp)

        self.use_graphnorm = use_graphnorm
        if use_graphnorm:
            self.out_norm = PyG_nn.GraphNorm(channels)

    @staticmethod
    def channel_shuffle(x, groups):
        N, C = x.shape
        assert C % groups == 0
        x = x.view(N, groups, -1)             # [N, G, C//G]
        x = x.permute(0, 2, 1).contiguous()   # [N, C//G, G]
        x = x.view(N, C)                      # flatten back
        return x

    def _sparsify(self, gates):
        if self.sparse_ratio <= 0.0 or not self.training:
            return gates
        k = max(1, int(gates.size(-1) * (1.0 - self.sparse_ratio)))
        values, indices = torch.topk(gates, k, dim=-1)
        threshold = values[..., -1].unsqueeze(-1)
        sparse_gates = torch.where(gates >= threshold, gates, torch.zeros_like(gates))
        return sparse_gates

    def forward(self, x, batch):

        N, C = x.shape
        G, Cg, H = self.groups, self.group_c, self.half_group_c
        assert C == self.channels
        xg = x.view(N, G, Cg)
        x_c = xg[:, :, :H].contiguous()
        x_s = xg[:, :, H:].contiguous()

        x_c_flat = x_c.view(N, -1)  # [N, G*H]
        gap_mean = PyG_nn.global_mean_pool(x_c_flat, batch)  # [B, G*H]
        batch_size = int(batch.max().item() + 1) if batch.numel() else 1

        device = x.device
        gap_max = x_c_flat.new_full((batch_size, x_c_flat.size(1)), float("-inf"))
        for i in range(batch_size):
            mask = (batch == i)
            if mask.any():
                gap_max[i] = x_c_flat[mask].max(dim=0).values
            else:
                gap_max[i] = 0.0
        gap_cat = torch.cat([gap_mean, gap_max], dim=1)  # [B, 2*G*H]

        ch_logits = self.channel_mlp(gap_cat)  # [B, G*H]
        ch_gate = self.sigmoid(ch_logits.view(-1, G, H) / self.temp)
        ch_gate = self.attn_dropout(ch_gate)
        ch_gate = ch_gate * (1.0 + self.cweight.view(1, 1, -1)) + self.cbias.view(1, 1, -1)
        ch_gate = self._sparsify(ch_gate)
        x_c = x_c * ch_gate[batch]

        xs_flat = x_s.view(N, G * H).unsqueeze(-1)  # [N, G*H, 1]
        xs_norm = self.gn(xs_flat).squeeze(-1).view(N, G, H)  # [N, G, H]
        mean_ctx = xs_norm.mean(dim=1)   # [N, H]
        max_ctx, _ = xs_norm.max(dim=1)  # [N, H]
        std_ctx = xs_norm.std(dim=1)     # [N, H]
        space_context = torch.cat([mean_ctx, max_ctx, std_ctx], dim=-1)  # [N, 3H]

        sp_logits = self.space_mlp(space_context)  # [N, H]
        sp_gate = self.sigmoid(sp_logits.view(N, 1, H) / self.temp)  # [N,1,H]
        sp_gate = self.attn_dropout(sp_gate)
        sp_gate = sp_gate * (1.0 + self.sweight.view(1, 1, -1)) + self.sbias.view(1, 1, -1)
        x_s = x_s * sp_gate

        out_stack = torch.stack([x_c, x_s], dim=-1)      # [N, G, H, 2]
        out_stack = out_stack.permute(0, 1, 3, 2).contiguous()  # [N, G, 2, H]
        out = out_stack.view(N, G * 2 * H)  # [N, C]

        gamma = torch.tanh(self.init_scale)
        out = out + gamma * x

        if self.use_graphnorm:
            out = self.out_norm(out, batch)
        out = self.channel_shuffle(out, groups=2)
        return out



class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNN, self).__init__()
        # 局部特征提取
        self.gnn1 = PyG.nn.SAGEConv(input_dim, 128)
        self.gnn2 = PyG.nn.SAGEConv(128, 256)

        # 局部后PGSCA
        self.pgsca_local = ParallelGraphSCAttention(256, groups=8, reduction=4, attn_dropout=0.1, sparse_ratio=0.2,
                                                    temp=0.5)

        # 全局交互
        self.transformer1 = PyG.nn.TransformerConv(256, 256, heads=1)
        self.transformer2 = PyG.nn.TransformerConv(256, 256, heads=1)

        # 全局后PGSCA
        self.pgsca_global = ParallelGraphSCAttention(256, groups=8, reduction=4, attn_dropout=0.1, sparse_ratio=0.0,
                                                     temp=1.0)

        # 投影层
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim)
        )

        # 注意力池化
        self.att_pool = PyG.nn.GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))

        # 融合参数
        self.fuse_alpha = nn.Parameter(torch.tensor(0.5))


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 局部GNN
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))

        # 局部PGSCA增强
        x_local = self.pgsca_local(x, batch)

        # 全局Transformer
        x = F.relu(self.transformer1(x_local, edge_index))  # 用增强x输入
        x = F.relu(self.transformer2(x, edge_index))

        # 全局PGSCA增强
        x_global = self.pgsca_global(x, batch)

        # 原分支: 投影 + max pool
        x_proj_orig = self.proj(x_global)
        g_orig = PyG.nn.global_max_pool(x_proj_orig, batch)  # [B, hidden_dim]

        # 增强分支: 投影 + att pool
        x_proj_enh = self.proj(x_global)
        g_enh = self.att_pool(x_proj_enh, batch)  # [B, hidden_dim]

        alpha = torch.sigmoid(self.fuse_alpha)
        g = g_orig + alpha * g_enh  # 或 g = (1 - alpha) * g_orig + alpha * g_enh

        return g

class PeptideBERT(torch.nn.Module):
    def __init__(self, bert_config):
        super(PeptideBERT, self).__init__()
        local_model_path = "/mnt/sdb/cjj/Mmodel/prot_bert_bfd"
        self.protbert = BertModel.from_pretrained(
            local_model_path,
            config=bert_config,
            ignore_mismatched_sizes=True
        )

    def forward(self, inputs, attention_mask):
        output = self.protbert(inputs, attention_mask=attention_mask)
        return output.pooler_output

class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class PretrainNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, bert_config, projection_dim, dropout):
        super(PretrainNetwork, self).__init__()
        self.gnn = GNN(input_dim, hidden_dim)
        self.bert = PeptideBERT(bert_config)
        self.graph_projection = ProjectionHead(hidden_dim, projection_dim, dropout)
        self.text_projection = ProjectionHead(bert_config.hidden_size, projection_dim, dropout)

    def forward(self, data):
        gnn_features = self.gnn(data)
        bert_features = self.bert(data.seq, data.attn_mask)

        gnn_embs = self.graph_projection(gnn_features)
        bert_embs = self.text_projection(bert_features)

        gnn_embs = gnn_embs / torch.linalg.norm(gnn_embs, dim=1, keepdim=True)
        bert_embs = bert_embs / torch.linalg.norm(bert_embs, dim=1, keepdim=True)

        return bert_embs, gnn_embs



class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, use_intra=True):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.use_intra = use_intra

    def forward(self, bert_embs, gnn_embs, labels):
        bert_embs = F.normalize(bert_embs, dim=1)
        gnn_embs = F.normalize(gnn_embs, dim=1)

        logits = bert_embs @ gnn_embs.T  # [B, B]
        logits = logits / self.temperature
        labels_inter = torch.arange(logits.size(0), device=logits.device)

        inter_loss_1 = F.cross_entropy(logits, labels_inter)
        inter_loss_2 = F.cross_entropy(logits.T, labels_inter)
        inter_loss = (inter_loss_1 + inter_loss_2) / 2

        if not self.use_intra:
            return inter_loss

        mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        intra_bert_loss = self.nt_xent_loss(bert_embs, mask)
        intra_gnn_loss = self.nt_xent_loss(gnn_embs, mask)
        intra_loss = (intra_bert_loss + intra_gnn_loss) / 2

        return (inter_loss + intra_loss) / 2

    def nt_xent_loss(self, embeddings, label_mask):
        B = embeddings.size(0)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature  # [B, B]

        logits_mask = ~torch.eye(B, dtype=torch.bool, device=embeddings.device)
        sim_matrix = sim_matrix.masked_fill(~logits_mask, float('-inf'))

        positive_mask = label_mask ^ torch.eye(B, dtype=torch.bool, device=embeddings.device)
        positive_mask = positive_mask & logits_mask

        exp_sim = torch.exp(sim_matrix)
        denom = exp_sim.sum(dim=1) + 1e-8
        pos_exp = exp_sim * positive_mask.float()
        pos_sum = pos_exp.sum(dim=1) + 1e-8

        loss = -torch.log(pos_sum / denom)
        return loss.mean()



def create_model(config, get_embeddings=False):
    bert_config = BertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['network']['BERT']['hidden_size'],
        num_hidden_layers=config['network']['BERT']['hidden_layers'],
        num_attention_heads=config['network']['BERT']['attn_heads'],
        hidden_dropout_prob=config['network']['BERT']['dropout']
    )

    model = PretrainNetwork(
        input_dim=config['network']['GNN']['input_dim'],
        hidden_dim=config['network']['GNN']['hidden_dim'],
        bert_config=bert_config,
        projection_dim=config['network']['proj_dim'],
        dropout=config['network']['drp']
    ).to(config['device'])

    return model

def cri_opt_sch(config, model):
    criterion = CLIPLoss(
        temperature=config["loss"]["temperature"],
        use_intra=config["loss"]["use_intra"]
    )


    for p in model.gnn.parameters():
        p.requires_grad = False

    for p in model.gnn.pgsca_local.parameters():
        p.requires_grad = True
    for p in model.gnn.pgsca_global.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": config["optim"]["lr_bert"]},
        {"params": model.text_projection.parameters(), "lr": config["optim"]["lr_bert"]},
        {"params": model.graph_projection.parameters(), "lr": config["optim"]["lr_gnn"]},
        {"params": model.gnn.pgsca_local.parameters(), "lr": 1e-5},
        {"params": model.gnn.pgsca_global.parameters(), "lr": 1e-5},
    ])


    if config["sch"]["name"] == "lronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["sch"]["factor"],
            patience=config["sch"]["patience"],
            min_lr=config["sch"].get("min_lr", 1e-6)
        )
    elif config["sch"]["name"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max(config["optim"]["lr_gnn"], config["optim"]["lr_bert"]),
            epochs=config["epochs"],
            steps_per_epoch=config["sch"]["steps"]
        )
    else:
        scheduler = None

    return criterion, optimizer, scheduler


class FinetuneNetwork(nn.Module):

    def __init__(self, backbone: PretrainNetwork, proj_dim: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone


        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, data):
        bert_embs, gnn_embs = self.backbone(data)   # [B, proj_dim], [B, proj_dim]
        feat = torch.cat([bert_embs, gnn_embs], dim=1)  # [B, 2*proj_dim]
        logits = self.classifier(feat).squeeze(-1)  # [B]
        return logits

def create_finetune_model(config):

    backbone = create_model(config)
    model = FinetuneNetwork(
        backbone=backbone,
        proj_dim=config["network"]["proj_dim"],
        dropout=config["network"]["drp"]
    ).to(config["device"])
    return model

def cri_opt_sch_finetune(config, model):

    # ===== loss =====
    pos_weight = config.get("finetune", {}).get("pos_weight", None)
    if pos_weight is not None:
        pos_weight = torch.tensor([float(pos_weight)], device=config["device"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ===== freeze strategy =====
    freeze_backbone = config.get("finetune", {}).get("freeze_backbone", True)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # ===== optimizer =====
    lr_head = config.get("finetune", {}).get("lr_head", 1e-3)
    lr_backbone = config.get("finetune", {}).get("lr_backbone", 1e-5)

    params = [{"params": model.classifier.parameters(), "lr": lr_head}]

    if not freeze_backbone:
        params += [{"params": model.backbone.parameters(), "lr": lr_backbone}]

    optimizer = torch.optim.AdamW(params, weight_decay=1e-2)

    # ===== scheduler =====
    sch_name = config.get("finetune",    {}).get("scheduler", "lronplateau")
    if sch_name == "lronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config["sch"]["factor"],
            patience=config["sch"]["patience"],
            min_lr=config["sch"].get("min_lr", 1e-6)
        )
    else:
        scheduler = None

    return criterion, optimizer, scheduler
