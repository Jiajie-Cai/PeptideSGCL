import torch
from tqdm import tqdm
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def find_best_threshold(y_true, y_prob, mode="mcc"):
    """
    扫描阈值 0~1，寻找最优阈值（默认 MCC 最大）。
    mode: "mcc" or "f1"
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_t, best_score = 0.5, -1e9
    best_metrics = None

    thresholds = np.linspace(0.0, 1.0, 101)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        # 手写 MCC，避免 sklearn 缺失时崩
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
        mcc = (tp * tn - fp * fn) / denom

        # F1
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        score = mcc if mode == "mcc" else f1
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_metrics = {
                "mcc": float(mcc),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
            }

    return best_t, best_metrics


def compute_metrics(y_true, y_logits, threshold_mode="mcc"):
    """
    输入 logits，输出常用二分类指标（AUC/MCC/F1/Acc/Precision/Recall + best_threshold）
    """
    y_true = np.asarray(y_true).astype(int)
    y_logits = np.asarray(y_logits).astype(float)
    y_prob = sigmoid(y_logits)

    # AUC（如果 sklearn 可用）
    auc = None
    if SKLEARN_OK:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None

    best_t, best_extra = find_best_threshold(y_true, y_prob, mode=threshold_mode)
    y_pred = (y_prob >= best_t).astype(int)

    # 其他指标（如果 sklearn 可用则用 sklearn）
    if SKLEARN_OK:
        mcc = float(matthews_corrcoef(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred))
        acc = float(accuracy_score(y_true, y_pred))
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
    else:
        # fallback：用 best_extra
        mcc = best_extra["mcc"]
        f1 = best_extra["f1"]
        precision = best_extra["precision"]
        recall = best_extra["recall"]
        acc = float((y_pred == y_true).mean())
        balanced_acc = acc  # fallback: same as acc when sklearn unavailable

    return {
        "auc": auc,
        "mcc": mcc,
        "f1": f1,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "precision": precision,
        "recall": recall,
        "best_threshold": best_t
    }


def train_cls(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    bar = tqdm(dataloader, desc="Train-CLS", leave=False, dynamic_ncols=True)
    total_loss = 0.0

    for i, batch in enumerate(bar):
        batch = batch.to(device)
        labels = batch.label.float()  # [B], 0/1

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)  # [B]
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_postfix(loss=f"{total_loss / (i + 1):.4f}")

    return total_loss / len(dataloader)


def validate_cls(model, dataloader, criterion, device, threshold_mode="mcc"):
    model.eval()
    bar = tqdm(dataloader, desc="Val-CLS", leave=False, dynamic_ncols=True)

    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.inference_mode():
        for i, batch in enumerate(bar):
            batch = batch.to(device)
            labels = batch.label.float()

            logits = model(batch)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            bar.set_postfix(loss=f"{total_loss / (i + 1):.4f}")

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = compute_metrics(all_labels, all_logits, threshold_mode=threshold_mode)

    return total_loss / len(dataloader), metrics
