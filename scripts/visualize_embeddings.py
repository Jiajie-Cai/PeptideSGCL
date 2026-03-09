import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import umap
from mpl_toolkits.mplot3d import Axes3D
import yaml

# -------------------------------
# 1️⃣ 模型与数据加载部分（根据你的 baseline 修改路径）
# -------------------------------

from model.network import create_model  # 你baseline的模型构造函数
from data.dataloader import load_data   # 你baseline的数据加载函数

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2️⃣ 特征提取函数：输出三种模态的嵌入
# -------------------------------
def extract_embeddings(model, dataloader):
    model.eval()
    all_bert, all_gnn, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            # 将batch数据移动到设备
            batch = batch.to(device)
            labels = batch.label.cpu().numpy()

            # 模型前向传播
            bert_embs, gnn_embs = model(batch)
            
            # 收集嵌入向量
            all_bert.append(bert_embs.cpu().numpy())
            all_gnn.append(gnn_embs.cpu().numpy())
            all_labels.append(labels)

    return (
        np.concatenate(all_bert, axis=0),
        np.concatenate(all_gnn, axis=0),
        np.concatenate(all_labels, axis=0),
    )

# -------------------------------
# 3️⃣ 降维 + 绘图函数（含2D/3D）
# -------------------------------
def reduce_and_plot(features, labels, title_prefix, method="tsne"):
    os.makedirs(os.path.dirname(f"results/visualizations/{title_prefix}"), exist_ok=True)
    method = method.lower()

    # 选择降维算法
    reducer_2d = TSNE(n_components=2, random_state=42) if method == "tsne" else umap.UMAP(n_components=2, random_state=42)
    reducer_3d = TSNE(n_components=3, random_state=42) if method == "tsne" else umap.UMAP(n_components=3, random_state=42)

    reduced_2d = reducer_2d.fit_transform(features)
    reduced_3d = reducer_3d.fit_transform(features)

    # ---- 2D ----
    plt.figure(figsize=(8, 6), dpi=300)
    sns.scatterplot(
        x=reduced_2d[:, 0], y=reduced_2d[:, 1],
        hue=labels, palette="coolwarm", s=15, alpha=0.8, edgecolor=None, legend="brief"
    )
    plt.title(f"{title_prefix.split('/')[-1]} - {method.upper()} 2D", fontsize=12)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.legend(title="Class", loc="best")
    plt.tight_layout()
    path_2d = f"results/visualizations/{title_prefix} - 2D.png"
    plt.savefig(path_2d, dpi=300)
    plt.close()
    print(f"✅ Saved: {path_2d}")

    # ---- 3D ----
    fig = plt.figure(figsize=(8, 6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2],
        c=labels, cmap="coolwarm", s=10, alpha=0.8
    )
    ax.set_title(f"{title_prefix.split('/')[-1]} - {method.upper()} 3D", fontsize=12)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.04, label="Class")
    plt.tight_layout()
    path_3d = f"results/visualizations/{title_prefix} - 3D.png"
    plt.savefig(path_3d, dpi=300)
    plt.close()
    print(f"✅ Saved: {path_3d}")

# -------------------------------
# 4️⃣ 主控制函数
# -------------------------------
def visualize_for_task(task, method="tsne"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1. 读 config
    # config_path = f'./checkpoints/CLIP_balanced_data/{task}/config.yaml'
    # # # if not os.path.exists(config_path):
    config_path = './config.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    config['device'] = device
    config['task']  = task

    # 2. 模型
    model = create_model(config)
    BERT_state_dict = torch.load(
        f'./checkpoints/individual_pretrained/balanced_data/BERT/{task}/model.pt',
        map_location=device
    )
    model.bert.load_state_dict(BERT_state_dict['bert_state_dict'], strict=False)
    for param in model.gnn.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    # 3. 数据
    _, val_data_loader = load_data(config)   # 训练集不需要

    # 4. 提取 + 可视化
    bert, gnn, labels = extract_embeddings(model, val_data_loader)
    os.makedirs(f"results/visualizations/{task}", exist_ok=True)
    reduce_and_plot(bert, labels, f"{task}/PeptideBERT Embeddings", method)
    reduce_and_plot(gnn, labels, f"{task}/GNN Embeddings", method)
# -------------------------------
# 5️⃣ 主程序入口：hemo + nf 两个任务
# -------------------------------
if __name__ == "__main__":
    for task in ["hemo", "nf"]:
        visualize_for_task(task, method="tsne")   # 可改为 "umap"
