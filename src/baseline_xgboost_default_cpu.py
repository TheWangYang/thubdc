from datetime import datetime
from xgboost import XGBClassifier
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# 定义需要的函数
def sScore(y_true, y_pred):
    score = []
    for i in range(num_classes):
        score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    return score


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
feature = pd.DataFrame(Parallel(n_jobs=16, backend="multiprocessing")(
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

# 创建一个全零矩阵ovr_oof，用于存储每个样本在交叉验证中的预测概率
ovr_oof = np.zeros((len(train_scaler_X), num_classes))
# 创建一个全零矩阵ovr_preds，用于存储测试集中每个样本的预测概率
ovr_preds = np.zeros((len(test_scaler_X), num_classes))


# 开始训练过程
for train_index, valid_index in kf.split(train_scaler_X, y):
    X_train, X_valid = train_scaler_X[train_index], train_scaler_X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    clf = OneVsRestClassifier(XGBClassifier(random_state=0, n_jobs=8))
    clf.fit(X_train, y_train)
    ovr_oof[valid_index] = clf.predict_proba(X_valid)
    ovr_preds += clf.predict_proba(test_scaler_X) / n_splits
    score = sScore(y_valid, ovr_oof[valid_index])
    print(f"Score = {np.mean(score)}")


each_score = sScore(y, ovr_oof)
score_metric = pd.DataFrame(
    each_score, columns=['score'], index=list(lb_encoder.classes_))
score_metric.loc["Weighted AVG.", "score"] = np.mean(score_metric['score'])
print(score_metric)

# 结果提交
submit = pd.DataFrame(ovr_preds, columns=lb_encoder.classes_)
submit.index = X[all_data['label'].isnull()].index
submit.reset_index(inplace=True)
submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_,
                     value_name="score", var_name="source")

# 获取当前时间
current_time = datetime.now().strftime("%Y%m%d%H%M")  # 格式化为年月日时分

# 构造文件名
file_name = f"baseline_xgboost_default_{current_time}.csv"

# 保存为CSV文件
submit.to_csv("/data1/wyy/projects/_competition/thubdc/results/" +
              file_name, index=False)
