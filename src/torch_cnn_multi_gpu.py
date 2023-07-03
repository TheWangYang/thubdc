import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed


warnings.filterwarnings('ignore')


class DeepNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


# Define the sScore function using PyTorch
def sScore(y_true, y_pred):
    score = []
    for i in range(num_classes):
        score.append(roc_auc_score(
            y_true[:, i], y_pred[:, i]))

    return score


# Define a custom dataset class
class CustomDataset(data.Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        else:
            return self.X[index]


def processing_feature(file):
    log, trace, metric, metric_df = pd.DataFrame(
    ), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if os.path.exists(f"./inputs/log/{file}_log.csv"):
        log = pd.read_csv(
            f"./inputs/log/{file}_log.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"./inputs/trace/{file}_trace.csv"):
        trace = pd.read_csv(
            f"./inputs/trace/{file}_trace.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"./inputs/metric/{file}_metric.csv"):
        metric = pd.read_csv(
            f"./inputs/metric/{file}_metric.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    feats = {"id": file}
    if len(trace) > 0:
        feats['trace_length'] = len(trace)
        feats[f"trace_status_code_std"] = trace['status_code'].apply("std")

        # 计算列的标准差
        for stats_func in ['mean', 'std', 'skew', 'nunique']:
            feats[f"trace_timestamp_{stats_func}"] = trace['timestamp'].apply(
                stats_func)

        for stats_func in ['nunique']:
            for i in ['host_ip', 'service_name', 'endpoint_name', 'trace_id', 'span_id', 'parent_id', 'start_time', 'end_time']:
                feats[f"trace_{i}_{stats_func}"] = trace[i].agg(stats_func)

    else:
        feats['trace_length'] = -1

    if len(log) > 0:
        feats['log_length'] = len(log)
        log['message_length'] = log['message'].fillna("").map(len)
        log['log_info_length'] = log['message'].map(
            lambda x: x.split("INFO")).map(len)

    else:
        feats['log_length'] = -1

    if len(metric) > 0:
        feats['metric_length'] = len(metric)
        feats['metric_value_timestamp_value_mean_std'] = metric.groupby(['timestamp'])[
            'value'].mean().std()

    else:
        feats['metric_length'] = -1

    return feats


def gen_label(train):
    col = np.zeros((train.shape[0], 9))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1

    return col


# -----------------------------------特征工程-------------------------------------
# 得到所有id的长度
all_ids = set([i.split("_")[0] for i in os.listdir("./inputs/metric/")]) |\
    set([i.split("_")[0] for i in os.listdir("./inputs/log/")]) |\
    set([i.split("_")[0] for i in os.listdir("./inputs/trace/")])
all_ids = list(all_ids)
print("IDs Length =", len(all_ids))

# 从所有的log/trace/metric文件夹中提取特征
feature = pd.DataFrame(Parallel(n_jobs=64, backend="multiprocessing")(
    delayed(processing_feature)(f) for f in tqdm(all_ids)))

# 获得所有训练标签
label = pd.read_csv("./labels/training_label.csv")
# 对标签进行编码
lb_encoder = LabelEncoder()
# 对标签中的source进行编码，并存储在label字典的'label'键对应的项目中
label['label'] = lb_encoder.fit_transform(label['source'])

# 将特征数据和标签数据进行合并，并按照'id'列进行连接。合并后的结果存储在all_data变量中，并将'id'列设置为索引列
all_data = feature.merge(label[['id', 'label']].groupby(['id'], as_index=False)[
                         'label'].agg(list), how='left', on=['id']).set_index("id")

# 设置不使用的列名
not_use = ['id', 'label']
# 将其他使用的列名都作为特征
feature_name = [i for i in all_data.columns if i not in not_use]
# 从all_data中提取特征列，并对其中的无穷大值和负无穷大值进行替换和剪裁操作
X = all_data[feature_name].replace([np.inf, -np.inf], 0).clip(-1e9, 1e9)
print(f"Feature Length = {len(feature_name)}")
print(f"Feature = {feature_name}")


# ----------------------------------训练过程---------------------------------------
num_classes = 9
n_splits = 5
kf = MultilabelStratifiedKFold(
    n_splits=n_splits, random_state=3407, shuffle=True)
# 用于特征缩放
scaler = StandardScaler()
# 对特征数据进行缺失值填充、替换和缩放操作，得到缩放后的特征矩阵scaler_X
scaler_X = scaler.fit_transform(X.fillna(0).replace([np.inf, -np.inf], 0))

# 得到标签
y = gen_label(all_data[all_data['label'].notnull()])
# 提取有标签的特征矩阵（训练集）
train_scaler_X = scaler_X[all_data['label'].notnull()]
# 提取无标签的特征矩阵（测试集）
test_scaler_X = scaler_X[all_data['label'].isnull()]


# --------------------------------------使用单机多卡训练代码------------------------------------------
# 设置超参数
input_dim = len(feature_name)
hidden_dim = 128
output_dim = num_classes
batch_size = 32
learning_rate = 0.001
num_epochs = 300

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化分布式训练环境
dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()
rank = dist.get_rank()

# 加载数据集
train_dataset = CustomDataset(torch.tensor(train_scaler_X, dtype=torch.float32),
                              torch.tensor(y, dtype=torch.float32))
train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler)

# 构建模型
net = DeepNet(input_dim, hidden_dim, output_dim).to(device)
net = DDP(net, device_ids=[rank])

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    net.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Reduce and average the losses across all processes
    loss_tensor = torch.tensor(total_loss, dtype=torch.float32, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    total_loss = loss_tensor.item() / world_size

    if rank == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# 在验证集（这里的验证集就是训练集）上进行预测
net.eval()
ovr_oof = np.zeros((len(train_scaler_X), num_classes))
with torch.no_grad():
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        ovr_oof[i * batch_size: (i + 1) *
                batch_size] = torch.sigmoid(outputs).detach().cpu().numpy()

# Calculate scores
each_score = sScore(y, ovr_oof)
score_metric = pd.DataFrame(
    each_score, columns=['score'], index=list(lb_encoder.classes_))
score_metric.loc["Weighted AVG.", "score"] = np.mean(score_metric['score'])
print(score_metric)

# Result submission
test_inputs = torch.tensor(test_scaler_X, dtype=torch.float32)
test_dataset = CustomDataset(test_inputs)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
ovr_preds = np.zeros((len(test_scaler_X), num_classes))

net.eval()
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        ovr_preds[i * batch_size: (i + 1) * batch_size] = torch.sigmoid(
            outputs).detach().cpu().numpy()

# 清理分布式训练环境
dist.destroy_process_group()

# Prepare submission dataframe
submit = pd.DataFrame(ovr_preds, columns=lb_encoder.classes_)
submit.index = X[all_data['label'].isnull()].index
submit.reset_index(inplace=True)
submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_,
                     value_name="score", var_name="source")

# Get current time
current_time = datetime.now().strftime("%Y%m%d%H%M")

# Construct the file name
file_name = f"torch_cnn_multi_gpu_{current_time}.csv"

# Save as a CSV file
submit.to_csv("/data1/wyy/projects/_competition/thubdc/results/" +
              file_name, index=False)
