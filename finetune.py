import os
import yaml
import torch
import wandb
import numpy as np
import random
from datetime import datetime

from data.dataloader import load_data
from model.network import create_finetune_model, cri_opt_sch_finetune
from model.utils_cls import train_cls, validate_cls
from model.early_stopping import EarlyStopping


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



def compute_pos_weight_from_loader(train_loader, device="cpu"):
    n_pos, n_neg = 0, 0

    for batch in train_loader:
        y = batch.label


        if isinstance(y, torch.Tensor):
            y = y.detach().cpu()

        n_pos += int((y == 1).sum())
        n_neg += int((y == 0).sum())

    pos_weight = n_neg / (n_pos + 1e-12)
    return pos_weight, n_pos, n_neg


def load_clip_backbone(model, clip_ckpt_path, device):
    ckpt = torch.load(clip_ckpt_path, map_location=device)

    model.backbone.gnn.load_state_dict(ckpt["gnn_state_dict"], strict=True)

    model.backbone.bert.protbert.load_state_dict(ckpt["bert_state_dict"], strict=False)

    print(f"✅ Loaded CLIP backbone from: {clip_ckpt_path}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    config["device"] = device

    config.setdefault("finetune", {})
    config["finetune"].setdefault("freeze_backbone", True)
    config["finetune"].setdefault("lr_head", 1e-3)
    config["finetune"].setdefault("lr_backbone", 1e-5)
    config["finetune"].setdefault("scheduler", "lronplateau")
    config["finetune"].setdefault("threshold_mode", "mcc")  # "mcc" or "f1"


    seed_everything(config.get("seed", 3407))

    # ====== data ======
    train_loader, val_loader = load_data(config)

    pos_weight, n_pos, n_neg = compute_pos_weight_from_loader(train_loader)
    print(f"Train set: pos={n_pos}, neg={n_neg}, pos_weight={pos_weight:.4f}")
    config["finetune"]["pos_weight"] = float(pos_weight)

    # ====== model ======
    model = create_finetune_model(config).to(device)

    # ====== load CLIP pretrained backbone ======
    clip_ckpt_path = f'./checkpoints/CLIP_balanced_data/{config["task"]}/model.pt'
    if os.path.exists(clip_ckpt_path):
        load_clip_backbone(model, clip_ckpt_path, device)
    else:
        print(f"⚠️ CLIP checkpoint not found: {clip_ckpt_path}")
        print("⚠️ Will finetune from random init (not recommended).")

    # ====== loss / optim / scheduler ======
    criterion, optimizer, scheduler = cri_opt_sch_finetune(config, model)

    # ====== wandb ======
    if not config["debug"]:
        run_name = f'finetune_{config["task"]}_{datetime.now().strftime("%m%d_%H%M")}'
        wandb.init(project="PeptideFold", name=run_name, config=config)

    # ====== save dir ======
    save_dir = f'./checkpoints/FINETUNE_{config["task"]}'
    os.makedirs(save_dir, exist_ok=True)

    early_stopper = EarlyStopping(patience=10, delta=1e-4, verbose=True)

    best_mcc = -1e9

    for epoch in range(config["epochs"]):
        train_loss = train_cls(model, train_loader, optimizer, criterion, scheduler, device)

        val_loss, metrics = validate_cls(
            model, val_loader, criterion, device,
            threshold_mode=config["finetune"]["threshold_mode"]
        )

        msg = (
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"TrainLoss={train_loss:.4f} ValLoss={val_loss:.4f} | "
            f"AUC={metrics['auc']} MCC={metrics['mcc']:.4f} "
            f"F1={metrics['f1']:.4f} Acc={metrics['acc']:.4f} Balanced_Acc={metrics['balanced_acc']:.4f} "
            f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} "
            f"BestT={metrics['best_threshold']:.2f}"
        )
        print(msg)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        if not config["debug"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val/{k}": v for k, v in metrics.items() if v is not None}
            })

        if metrics["mcc"] > best_mcc:
            best_mcc = metrics["mcc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"✅ Best model saved! MCC={best_mcc:.4f}\n")


        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

    if not config["debug"]:
        wandb.finish()


if __name__ == "__main__":
    main()
