import os
import shutil
import random
import numpy as np
import torch
from datetime import datetime
import yaml
import wandb

from data.dataloader import load_data
from model.network import create_model, cri_opt_sch
from model.utils import train, validate
from model.early_stopping import EarlyStopping


def seed_everything(seed: int):
    """固定所有随机种子以确保结果可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


def train_model(config, model, train_data_loader, val_data_loader, optimizer, criterion, scheduler, save_dir):
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    best_val_loss = float("inf")
    early_stopper = EarlyStopping(
        patience=config.get("early_stopping", {}).get("patience", 10),
        delta=config.get("early_stopping", {}).get("delta", 0.001),
        verbose=True
    )

    for epoch in range(config["epochs"]):
        train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss:.4f} \tLR: {curr_lr:.3e}')

        val_loss = validate(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss:.4f}\n')

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)


        if not config["debug"]:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": curr_lr
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "model.pt")
            torch.save({
                "epoch": epoch,
                "gnn_state_dict": model.gnn.state_dict(),
                "bert_state_dict": model.bert.protbert.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": curr_lr
            }, ckpt_path)
            print(f"✅ Model Saved to {ckpt_path}\n")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("⏹️ Early stopping triggered.")
            break

    return "Training completed"


def main():
    # ====== load config ======
    config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
    config["device"] = device


    seed_everything(config.get("seed", 3407))

    # ====== load data ======
    train_data_loader, val_data_loader = load_data(config)
    config["sch"]["steps"] = len(train_data_loader)

    # ====== model ======
    model = create_model(config).to(device)

    # ====== load pretrained bert (可选) ======
    bert_ckpt_path = f'./checkpoints/individual_pretrained/balanced_data/BERT/{config["task"]}/model.pt'
    if os.path.exists(bert_ckpt_path):
        bert_state = torch.load(bert_ckpt_path, map_location=device)

        model.bert.protbert.load_state_dict(bert_state["bert_state_dict"], strict=False)
        print(f"✅ Loaded pretrained BERT from {bert_ckpt_path}\n")
    else:
        print(f"⚠️ Pretrained BERT checkpoint not found: {bert_ckpt_path}\n")

    # ====== criterion / optim / scheduler ======
    criterion, optimizer, scheduler = cri_opt_sch(config, model)

    # ====== dirs & snapshot ======
    temp_dir = "./checkpoints/temp"
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy("./config.yaml", f"{temp_dir}/config.yaml")
    shutil.copy("./model/network.py", f"{temp_dir}/network.py")

    save_dir = f'./checkpoints/CLIP_balanced_data/{config["task"]}'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy("./config.yaml", f"{save_dir}/config.yaml")
    shutil.copy("./model/network.py", f"{save_dir}/network.py")

    # ====== wandb ======
    if not config["debug"]:
        run_name = f'c{datetime.now().strftime("%m%d_%H%M")}'
        wandb.init(project="PeptideFold", name=run_name, config=config)

    # ====== train ======
    train_model(config, model, train_data_loader, val_data_loader, optimizer, criterion, scheduler, save_dir)

    if not config["debug"]:
        wandb.finish()


if __name__ == "__main__":
    main()
