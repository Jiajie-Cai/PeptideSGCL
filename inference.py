# 程序最开始设置环境变量（确保哈希种子生效）
# import os
# os.environ['PYTHONHASHSEED'] = '19'  # 先临时设置，后续从config读取
# # 导入必要库
# import random
# import torch
# import yaml
# from tqdm import tqdm
# import numpy as np
# from data.dataloader import load_data
# from model.network import create_model
# from sklearn.metrics import (
#     accuracy_score, balanced_accuracy_score, precision_score, recall_score,
#     f1_score, roc_auc_score, average_precision_score, matthews_corrcoef
# )
#
#
# # 定义种子固定函数（与训练代码保持一致）
# def seed_everything(seed: int):
#     """固定所有随机种子，确保与训练过程一致"""
#     # 1. Python内置random
#     random.seed(seed)
#
#     # 2. 哈希种子（强化设置）
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     # 3. NumPy
#     np.random.seed(seed)
#
#     # 4. PyTorch CPU
#     torch.manual_seed(seed)
#
#     # 5. PyTorch GPU
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # 多GPU场景
#
#     # 6. cuDNN确定性设置
#     torch.backends.cudnn.deterministic = True  # 强制使用确定性算法
#     torch.backends.cudnn.benchmark = False  # 关闭自动优化（可能影响速度，但确保复现）
#
#
# def get_train_embeddings(dataloader, model, device):
#     gnn_embeddings, bert_embeddings = [], []
#     labels = []
#
#     for batch in tqdm(dataloader, leave=False):
#         batch = batch.to(device)
#         output = model(batch)
#
#         bert_embeddings.append(output[0].detach().cpu().numpy())
#         gnn_embeddings.append(output[1].detach().cpu().numpy())
#         labels.extend(batch.label.cpu().numpy())
#
#     return np.vstack(bert_embeddings).T, np.vstack(gnn_embeddings).T, labels
#
#
# def compute_metrics(y_true, y_pred, y_prob=None):
#     """计算多种评估指标"""
#     metrics = {}
#
#     # 基础指标
#     metrics["ACC"] = round(accuracy_score(y_true, y_pred), 4)
#     metrics["Balanced_ACC"] = round(balanced_accuracy_score(y_true, y_pred), 4)
#
#     # 精确率、召回率、F1
#     try:
#         metrics["Precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
#         metrics["Recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
#         metrics["F1_Score"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
#     except Exception as e:
#         print(f"精确率/召回率/F1计算异常: {e}")
#         metrics["Precision"] = metrics["Recall"] = metrics["F1_Score"] = float('nan')
#
#     # MCC
#     try:
#         metrics["MCC"] = round(matthews_corrcoef(y_true, y_pred), 4)
#     except Exception as e:
#         print(f"MCC计算异常: {e}")
#         metrics["MCC"] = float('nan')
#
#     # AUC相关指标 (需要概率值)
#     if y_prob is not None:
#         try:
#             metrics["AUC_ROC"] = round(roc_auc_score(y_true, y_prob), 4)
#         except Exception as e:
#             print(f"AUC_ROC计算异常: {e}")
#             metrics["AUC_ROC"] = float('nan')
#
#         try:
#             metrics["AUC_PR"] = round(average_precision_score(y_true, y_prob), 4)
#         except Exception as e:
#             print(f"AUC_PR计算异常: {e}")
#             metrics["AUC_PR"] = float('nan')
#
#     return metrics
#
#
# def main(train_data_loader, val_data_loader, model, device):
#     bert_embeddings, gnn_embeddings, labels = get_train_embeddings(train_data_loader, model, device)
#
#     bert_labels, gnn_labels = [], []
#     bert_probs, gnn_probs = [], []
#     ground_truth = []
#
#     for batch in tqdm(val_data_loader, leave=False):
#         batch = batch.to(device)
#         output = model(batch)
#
#         bert_output = output[0].detach().cpu().numpy()
#         gnn_output = output[1].detach().cpu().numpy()
#
#         # 计算相似度得分
#         bert_similarity = bert_output @ bert_embeddings
#         gnn_similarity = gnn_output @ gnn_embeddings
#
#         # 获取预测标签
#         bert_pred = np.argmax(bert_similarity, axis=1)
#         gnn_pred = np.argmax(gnn_similarity, axis=1)
#
#         bert_label = list(map(labels.__getitem__, bert_pred))
#         gnn_label = list(map(labels.__getitem__, gnn_pred))
#
#         # 获取正类的概率 (用于AUC计算)
#         bert_sim_softmax = np.exp(bert_similarity) / np.sum(np.exp(bert_similarity), axis=1, keepdims=True)
#         gnn_sim_softmax = np.exp(gnn_similarity) / np.sum(np.exp(gnn_similarity), axis=1, keepdims=True)
#
#         bert_prob = np.array(
#             [bert_sim_softmax[i, np.where(np.array(labels) == 1)[0]].sum() for i in range(len(bert_sim_softmax))])
#         gnn_prob = np.array(
#             [gnn_sim_softmax[i, np.where(np.array(labels) == 1)[0]].sum() for i in range(len(gnn_sim_softmax))])
#
#         bert_labels.extend(bert_label)
#         gnn_labels.extend(gnn_label)
#         bert_probs.extend(bert_prob)
#         gnn_probs.extend(gnn_prob)
#         ground_truth.extend(batch.label.cpu().numpy())
#
#     # 转换为numpy数组
#     bert_labels = np.array(bert_labels)
#     gnn_labels = np.array(gnn_labels)
#     bert_probs = np.array(bert_probs)
#     gnn_probs = np.array(gnn_probs)
#     ground_truth = np.array(ground_truth)
#
#     # 计算BERT模型的各项指标
#     print("\n" + "=" * 50)
#     print("BERT模型评估指标:")
#     print("=" * 50)
#     bert_metrics = compute_metrics(ground_truth, bert_labels, bert_probs)
#     for metric_name, metric_value in bert_metrics.items():
#         print(f"{metric_name}: {metric_value}")
#
#     # 计算GNN模型的各项指标
#     print("\n" + "=" * 50)
#     print("GNN模型评估指标:")
#     print("=" * 50)
#     gnn_metrics = compute_metrics(ground_truth, gnn_labels, gnn_probs)
#     for metric_name, metric_value in gnn_metrics.items():
#         print(f"{metric_name}: {metric_value}")
#
#     # 返回BERT的准确率作为主要指标
#     return bert_metrics["ACC"]
#
#
# if __name__ == '__main__':
#     # 1. 加载配置（优先从配置文件获取种子，与训练保持一致）
#     config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
#
#     # 2. 固定种子（核心：使用与训练相同的种子值）
#     seed = config.get('seed', 115514)
#     seed_everything(seed)
#     print(f"验证代码已固定所有种子，种子值: {seed}")
#
#     # 3. 设备设置
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(f'Device: {device}\n')
#     config['device'] = device
#
#     # 4. 加载数据（种子固定后再加载，确保数据划分一致）
#     train_data_loader, val_data_loader = load_data(config)
#
#     # 5. 初始化模型并加载权重
#     model = create_model(config)
#     model.eval()  # 验证模式
#
#     # 加载训练好的模型权重
#     BERT_state_dict = torch.load(f'./checkpoints/{config["task"]}/model.pt', map_location=device)
#     model.bert.load_state_dict(BERT_state_dict['bert_state_dict'], strict=False)
#
#     # 6. 执行验证
#     main(train_data_loader, val_data_loader, model, device)

import random
import torch
import yaml
import os
from tqdm import tqdm
import numpy as np
from data.dataloader import load_data
from model.network import create_model
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, average_precision_score, matthews_corrcoef
from scipy.special import softmax
# 新增：种子固定函数
def seed_everything(seed: int):
    """
    固定所有随机种子以确保结果可复现。
    """
    # 1. 固定 Python 内置的 random 模块
    random.seed(seed)

    # 2. 固定 os.environ['PYTHONHASHSEED']
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 3. 固定 NumPy
    np.random.seed(seed)

    # 4. 固定 PyTorch (CPU)
    torch.manual_seed(seed)

    # 5. 固定 PyTorch (GPU, if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前 GPU 设置种子
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置种子 (在 DDP 中很重要)

    # 6. 固定 cuDNN 的行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动寻找最优卷积算法，确保确定性


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "ACC": round(acc, 4),
        "Balanced_ACC": round(balanced_acc, 4),
        "Pre": round(pre, 4),
        "F1_Score": round(f1, 4),
        "Recall": round(recall, 4),
        "AUC_ROC": round(auc_roc, 4),
        "AUC_PR": round(auc_pr, 4),
        "MCC": round(mcc, 4)
    }


def get_train_embeddings(dataloader, model, device):
    gnn_embeddings, bert_embeddings = [], []
    labels = []

    for batch in tqdm(dataloader, leave=False):
        batch = batch.to(device)
        output = model(batch)

        bert_embeddings.append(output[0].detach().cpu().numpy())
        gnn_embeddings.append(output[1].detach().cpu().numpy())
        labels.extend(batch.label.cpu().numpy())

    return np.vstack(bert_embeddings).T, np.vstack(gnn_embeddings).T, labels


def main(train_data_loader, val_data_loader, model, device):
    bert_embeddings, gnn_embeddings, labels = get_train_embeddings(train_data_loader, model, device)

    train_labels = np.array(labels)  # Convert to array for easier indexing

    bert_labels, gnn_labels = [], []
    bert_y_probs, gnn_y_probs = [], []
    ground_truth = []
    num_correctb, num_correctg = 0, 0

    for batch in tqdm(val_data_loader, leave=False):
        batch = batch.to(device)
        output = model(batch)

        bert_output = output[0].detach().cpu().numpy()
        gnn_output = output[1].detach().cpu().numpy()

        bert_sim = bert_output @ bert_embeddings
        gnn_sim = gnn_output @ gnn_embeddings

        bert_pred = np.argmax(bert_sim, axis=1)
        gnn_pred = np.argmax(gnn_sim, axis=1)

        bert_label = [labels[i] for i in bert_pred]
        gnn_label = [labels[i] for i in gnn_pred]

        batch_labels = batch.label.cpu().numpy()
        num_correctb += np.sum(np.array(bert_label) == batch_labels)
        num_correctg += np.sum(np.array(gnn_label) == batch_labels)

        # Compute y_probs using softmax over similarities
        batch_size = len(bert_sim)
        for i in range(batch_size):
            bert_probs = softmax(bert_sim[i])
            gnn_probs = softmax(gnn_sim[i])

            bert_pos_prob = np.sum(bert_probs[train_labels == 1])
            gnn_pos_prob = np.sum(gnn_probs[train_labels == 1])

            bert_y_probs.append(bert_pos_prob)
            gnn_y_probs.append(gnn_pos_prob)

        bert_labels.extend(bert_label)
        gnn_labels.extend(gnn_label)
        ground_truth.extend(batch_labels)

    total_samples = len(val_data_loader.dataset)
    print(f'BERT Accuracy: {num_correctb / total_samples}')
    print(f'GNN Accuracy: {num_correctg / total_samples}')

    # Convert to arrays
    y_true = np.array(ground_truth)
    bert_y_pred = np.array(bert_labels)
    gnn_y_pred = np.array(gnn_labels)
    bert_y_prob = np.array(bert_y_probs)
    gnn_y_prob = np.array(gnn_y_probs)

    # Compute metrics
    bert_metrics = compute_metrics(y_true, bert_y_pred, bert_y_prob)
    gnn_metrics = compute_metrics(y_true, gnn_y_pred, gnn_y_prob)

    print('BERT Metrics:')
    for key, value in bert_metrics.items():
        print(f'{key}: {value}')

    print('GNN Metrics:')
    for key, value in gnn_metrics.items():
        print(f'{key}: {value}')

    return num_correctb / total_samples


if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
    config['device'] = device

    train_data_loader, val_data_loader = load_data(config)

    model = create_model(config)
    model.eval()

    # BERT_state_dict = torch.load(f'./checkpoints/updated_inference/hemo/model.pt', map_location=device)
    BERT_state_dict = torch.load(f'./checkpoints/{config["task"]}/model.pt', map_location=device)

    model.bert.load_state_dict(BERT_state_dict['bert_state_dict'], strict=False)
    # 改动2：原为“model.bert.load_state_dict(BERT_state_dict['model_state_dict'], strict=False)”，无法加载'model_state_dict'，但作者训练好的模型文件可以使用
    main(train_data_loader, val_data_loader, model, device)
