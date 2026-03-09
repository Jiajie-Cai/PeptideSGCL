import os
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader as GeoDataLoader

from data.dataset import PeptideFoldDataset
from model.network import create_model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
out_root = "explainability_outputs/embedding_c_experiments"
os.makedirs(out_root, exist_ok=True)

CHECKPOINTS = [
    ("intra_seed3407", "models/h0model.pth"),
    ("nointra_seed3407", "models/h5model.pth"),
]

BATCH_SIZE = 32


def load_config():
    """你的 config 统一从 config.py 或 yaml 加载，我按照 MultiPeptide 默认写："""
    import yaml
    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    config['device'] = device
    return config


def build_model_from_ckpt(ckpt_path, config):
    model = create_model(config)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    return model


def load_test_loader(config):
    data_path = config['paths']['data'] + config['task'] + '/'
    split_path = data_path + 'splits/' + config['paths']['split']

    test_dataset = PeptideFoldDataset(
        split_path + 'val.pkl',
        data_path + 'mapping_unnorm_11.pkl'
    )

    test_loader = GeoDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


def extract_and_save(label, ckpt_path, config):
    model = build_model_from_ckpt(ckpt_path, config)
    loader = load_test_loader(config)

    bert_list, gnn_list, ys = [], [], []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"Extract {label}"):
            data = data.to(device)

            bert_embs, gnn_embs = model(data)

            bert_list.append(bert_embs.cpu().numpy())
            gnn_list.append(gnn_embs.cpu().numpy())
            ys.append(data.y.cpu().numpy())

    bert_arr = np.vstack(bert_list)
    gnn_arr = np.vstack(gnn_list)
    y_arr = np.concatenate(ys)

    out_path = os.path.join(out_root, f"embeddings_{label}.npz")
    np.savez(out_path, bert=bert_arr, gnn=gnn_arr, y=y_arr)
    print("Saved", out_path)


if __name__ == "__main__":
    config = load_config()

    for label, ckpt in CHECKPOINTS:
        extract_and_save(label, ckpt, config)
