import os
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


try:
    import umap
    UMAP_OK = True
except Exception:
    UMAP_OK = False

from data.dataloader import load_data
from model.network import create_finetune_model



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_2d_scatter(Z2, y, title, save_path, s=10, alpha=0.55):
    y = y.astype(int)
    pos = (y == 1)
    neg = (y == 0)

    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[neg, 0], Z2[neg, 1], s=s, alpha=alpha, label="Negative (0)")
    plt.scatter(Z2[pos, 0], Z2[pos, 1], s=s, alpha=alpha, label="Positive (1)")

    c_neg = Z2[neg].mean(axis=0)
    c_pos = Z2[pos].mean(axis=0)
    plt.scatter([c_neg[0]], [c_neg[1]], s=120, marker="X", label="Center Neg")
    plt.scatter([c_pos[0]], [c_pos[1]], s=120, marker="X", label="Center Pos")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_hist_compare(pos_arr, neg_arr, title, xlabel, save_path, bins=50):
    plt.figure(figsize=(6, 4))
    plt.hist(neg_arr, bins=bins, alpha=0.6, density=True, label="Negative (0)")
    plt.hist(pos_arr, bins=bins, alpha=0.6, density=True, label="Positive (1)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def class_separation_stats(feat, y):

    y = y.astype(int)
    pos = feat[y == 1]
    neg = feat[y == 0]

    mu_pos = pos.mean(axis=0)
    mu_neg = neg.mean(axis=0)

    between = np.linalg.norm(mu_pos - mu_neg)

    within_pos = np.mean(np.linalg.norm(pos - mu_pos, axis=1))
    within_neg = np.mean(np.linalg.norm(neg - mu_neg, axis=1))
    within = 0.5 * (within_pos + within_neg)

    fisher = between / (within + 1e-12)

    return {
        "between_center_distance": float(between),
        "within_avg_distance": float(within),
        "fisher_ratio": float(fisher),
        "n_pos": int(pos.shape[0]),
        "n_neg": int(neg.shape[0]),
    }


@torch.inference_mode()
def extract_all_stages(model, dataloader, device, max_batches=None):
    model.eval()

    all_bert, all_gnn, all_fused, all_head1, all_logits, all_y = [], [], [], [], [], []

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        batch = batch.to(device)
        y = batch.label.float()

        # backbone 输出 embedding
        bert_embs, gnn_embs = model.backbone(batch)                 # [B,256], [B,256]
        fused = torch.cat([bert_embs, gnn_embs], dim=1)             # [B,512]

        # ✅ 分类头第一层 Linear 输出（关键）
        head_linear1 = model.classifier[0](fused)                   # [B,256]

        # logit
        logits = model.classifier(fused).squeeze(-1)                # [B]

        all_bert.append(bert_embs)
        all_gnn.append(gnn_embs)
        all_fused.append(fused)
        all_head1.append(head_linear1)
        all_logits.append(logits)
        all_y.append(y)

    bert = torch.cat(all_bert, dim=0)
    gnn = torch.cat(all_gnn, dim=0)
    fused = torch.cat(all_fused, dim=0)
    head1 = torch.cat(all_head1, dim=0)
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    logits_np = to_numpy(logits)
    y_np = to_numpy(y).astype(int)

    return {
        "bert": to_numpy(bert),
        "gnn": to_numpy(gnn),
        "fused": to_numpy(fused),
        "head_linear1": to_numpy(head1),     # ✅ 新增
        "logits": logits_np,
        "prob": sigmoid_np(logits_np),
        "y": y_np
    }


def visualize_embeddings(name, X, y, outdir):
    """
    对 bert/gnn/fused 三个阶段做 PCA / TSNE / UMAP
    """
    # ===== PCA =====
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum()
    plot_2d_scatter(
        Z_pca, y,
        title=f"PCA-2D ({name}) | explained_var={var:.3f}",
        save_path=os.path.join(outdir, f"PCA2D_{name}.png")
    )


    X_50 = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
    tsne = TSNE(
        n_components=2,
        random_state=0,
        perplexity=30,
        learning_rate="auto",
        init="pca"
    )
    Z_tsne = tsne.fit_transform(X_50)
    plot_2d_scatter(
        Z_tsne, y,
        title=f"t-SNE-2D ({name})",
        save_path=os.path.join(outdir, f"TSNE2D_{name}.png")
    )

    # ===== UMAP =====
    if UMAP_OK:
        reducer = umap.UMAP(
            n_components=2,
            random_state=0,
            n_neighbors=25,
            min_dist=0.1
        )
        Z_umap = reducer.fit_transform(X)
        plot_2d_scatter(
            Z_umap, y,
            title=f"UMAP-2D ({name})",
            save_path=os.path.join(outdir, f"UMAP2D_{name}.png")
        )
    else:
        print("[WARN] umap-learn not installed, skip UMAP plots.")


def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    config["device"] = device

    # 数据
    train_loader, val_loader = load_data(config)

    # 加载微调模型
    model = create_finetune_model(config).to(device)

    finetune_ckpt = f'./checkpoints/FINETUNE_{config["task"]}/best_model.pt'
    if not os.path.exists(finetune_ckpt):
        raise FileNotFoundError(f"Cannot find finetune checkpoint: {finetune_ckpt}")

    ckpt = torch.load(finetune_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"✅ Loaded finetuned model from: {finetune_ckpt}")

    outdir = "./feature_viz_val"
    ensure_dir(outdir)


    feats = extract_all_stages(model, val_loader, device=device, max_batches=None)
    y = feats["y"]

    # 1) Embedding 三阶段：PCA / t-SNE / UMAP
    visualize_embeddings("bert", feats["bert"], y, outdir)
    visualize_embeddings("gnn", feats["gnn"], y, outdir)
    visualize_embeddings("fused", feats["fused"], y, outdir)
    visualize_embeddings("head_linear1", feats["head_linear1"], y, outdir)


    logits = feats["logits"]
    prob = feats["prob"]

    logits_pos = logits[y == 1]
    logits_neg = logits[y == 0]
    prob_pos = prob[y == 1]
    prob_neg = prob[y == 0]

    plot_hist_compare(
        pos_arr=logits_pos,
        neg_arr=logits_neg,
        title="Logit Distribution (Positive vs Negative)",
        xlabel="logit",
        save_path=os.path.join(outdir, "logit_distribution.png"),
        bins=55
    )

    plot_hist_compare(
        pos_arr=prob_pos,
        neg_arr=prob_neg,
        title="Probability Distribution (Positive vs Negative)",
        xlabel="probability",
        save_path=os.path.join(outdir, "prob_distribution.png"),
        bins=55
    )


    summary_path = os.path.join(outdir, "summary_stats.txt")
    with open(summary_path, "w") as f:
        for stage in ["bert", "gnn", "fused", "head_linear1"]:
            stats = class_separation_stats(feats[stage], y)
            f.write(f"[{stage}]\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

        # logit/prob 基本统计
        f.write("[logits]\n")
        f.write(f"logits_pos mean={logits_pos.mean():.4f}, std={logits_pos.std():.4f}\n")
        f.write(f"logits_neg mean={logits_neg.mean():.4f}, std={logits_neg.std():.4f}\n\n")

        f.write("[prob]\n")
        f.write(f"prob_pos mean={prob_pos.mean():.4f}, std={prob_pos.std():.4f}\n")
        f.write(f"prob_neg mean={prob_neg.mean():.4f}, std={prob_neg.std():.4f}\n")

    print(f"✅ All figures saved to: {outdir}")
    print(f"✅ Summary stats saved to: {summary_path}")


if __name__ == "__main__":
    main()
