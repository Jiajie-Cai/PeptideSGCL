import torch
from torch_geometric.data import Data
import pickle
from tqdm import tqdm
from torchdrug.data import Graph


class PeptideFoldDataset:
    def __init__(self, split_path, map_path):
        self.data = []

        with open(split_path, 'rb') as f:
            keys = pickle.load(f)

        with open(map_path, 'rb') as f:
            mapping = pickle.load(f)

        for key in tqdm(keys):
            label = 0 if key.split('_')[1][0] == 'n' else 1

            self.data.append(Data(
                x=torch.tensor(mapping[key]['x'], dtype=torch.float),
                edge_index=torch.tensor(mapping[key]['edges'], dtype=torch.long),
                label=torch.tensor(label, dtype=torch.float),
                seq=torch.tensor(mapping[key]['seq'], dtype=torch.long).unsqueeze(0),
                attn_mask=torch.tensor(mapping[key]['seq'] > 0, dtype=torch.long).unsqueeze(0)
            ))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




