from torch_geometric.loader import DataLoader
from data.dataset import PeptideFoldDataset


def load_data(config):
    print(f'{"="*30}{"DATA":^20}{"="*30}')

    data_path = config['paths']['data'] + config['task'] + '/'
    split_path = data_path + 'splits/' + config['paths']['split']

    train_dataset = PeptideFoldDataset(
        split_path+'train.pkl',
        data_path+'mapping_unnorm_11.pkl'
    )
    val_dataset = PeptideFoldDataset(
        split_path+'val.pkl',
        data_path+'mapping_unnorm_11.pkl'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    print('Batch size:', config['batch_size'])

    print('Train dataset samples: ', len(train_dataset))
    print('Validation dataset samples: ', len(val_dataset))

    print('Train dataset batches: ', len(train_loader))
    print('Validation dataset batches: ', len(val_loader))

    print()

    return train_loader, val_loader

