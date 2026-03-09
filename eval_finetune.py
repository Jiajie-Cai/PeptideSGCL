import os
import yaml
import torch
import numpy as np
import random

from data.dataloader import load_data
from model.network import create_finetune_model, cri_opt_sch_finetune
from model.utils_cls import validate_cls


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight_from_loader(train_loader):
    """pos_weight = neg / pos"""
    n_pos, n_neg = 0, 0
    for batch in train_loader:
        y = batch.label
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()
        n_pos += int((y == 1).sum())
        n_neg += int((y == 0).sum())
    pos_weight = n_neg / (n_pos + 1e-12)
    return float(pos_weight), int(n_pos), int(n_neg)


def load_finetuned_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"✅ Loaded finetuned model from: {ckpt_path}")

    if "metrics" in ckpt:
        print("Saved metrics:", ckpt["metrics"])

    return ckpt


def main():
    # ===== device =====
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ===== config =====
    config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    config["device"] = device


    config.setdefault("finetune", {})
    config["finetune"].setdefault("freeze_backbone", True)
    config["finetune"].setdefault("lr_head", 1e-3)
    config["finetune"].setdefault("lr_backbone", 1e-5)
    config["finetune"].setdefault("scheduler", "lronplateau")
    config["finetune"].setdefault("threshold_mode", "mcc")

    seed_everything(config.get("seed", 3407))

    # ===== data =====
    train_loader, val_loader = load_data(config)


    pos_weight, n_pos, n_neg = compute_pos_weight_from_loader(train_loader)
    config["finetune"]["pos_weight"] = pos_weight
    print(f"Train set: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight:.4f}\n")

    # ===== model   =====
    model = create_finetune_model(config).to(device)

    # ===== load finetuned best model =====
    ckpt_path = f'./checkpoints/FINETUNE_{config["task"]}/best_model.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Cannot find checkpoint: {ckpt_path}")

    load_finetuned_checkpoint(model, ckpt_path, device)


    criterion, _, _ = cri_opt_sch_finetune(config, model)

    # ===== validate =====
    val_loss, metrics = validate_cls(
        model, val_loader, criterion, device,
        threshold_mode=config["finetune"]["threshold_mode"]
    )

    print("========== Evaluation on Validation Set ==========")
    print(f"ValLoss = {val_loss:.4f}")
    print(f"AUC     = {metrics['auc']}")
    print(f"MCC     = {metrics['mcc']:.4f}")
    print(f"F1      = {metrics['f1']:.4f}")
    print(f"Acc     = {metrics['acc']:.4f}")
    print(f"Balanced_Acc = {metrics['balanced_acc']:.4f}")
    print(f"Prec    = {metrics['precision']:.4f}")
    print(f"Recall  = {metrics['recall']:.4f}")
    print(f"BestT   = {metrics['best_threshold']:.2f}")
    print("=================================================")


if __name__ == "__main__":
    main()
