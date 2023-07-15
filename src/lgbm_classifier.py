from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
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
import lightgbm as lgb
import sys
import csv
from io import StringIO
from datetime import datetime

# # 创建一个StringIO对象来捕获控制台输出
# output_buffer = StringIO()
# sys.stdout = output_buffer

warnings.filterwarnings('ignore')

# 定义需要的函数


def sScore(y_true, y_pred):
    score = []
    for i in range(num_classes):
        score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    return score

# 处理特征


def processing_feature(file):
    log, trace, metric, metric_df = pd.DataFrame(
    ), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if os.path.exists(f"/data1/wyy/projects/_competition/thubdc/inputs/log/{file}_log.csv"):
        log = pd.read_csv(
            f"/data1/wyy/projects/_competition/thubdc/inputs/log/{file}_log.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"/data1/wyy/projects/_competition/thubdc/inputs/trace/{file}_trace.csv"):
        trace = pd.read_csv(
            f"/data1/wyy/projects/_competition/thubdc/inputs/trace/{file}_trace.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"/data1/wyy/projects/_competition/thubdc/inputs/metric/{file}_metric.csv"):
        metric = pd.read_csv(
            f"/data1/wyy/projects/_competition/thubdc/inputs/metric/{file}_metric.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    feats = {"id": file}

    # trace特征
    if len(trace) > 0:
        feats['trace_length'] = len(trace)
        feats[f"trace_status_code_std"] = trace['status_code'].apply("std")

        # 创造timestamp的统计特征
        # ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']
        # ['mean', 'std', 'skew', 'nunique']

        for stats_func in ['mean', 'std', 'skew', 'nunique']:
            feats[f"trace_timestamp_{stats_func}"] = trace['timestamp'].apply(
                stats_func)

        # 创造start_time的统计特征
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
            feats[f"trace_start_time_{stats_func}"] = trace['start_time'].apply(
                stats_func)

        # 创造end_time的统计特征
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
            feats[f"trace_end_time_{stats_func}"] = trace['end_time'].apply(
                stats_func)

        for stats_func in ['nunique']:
            for i in ['host_ip', 'service_name', 'endpoint_name', 'trace_id', 'span_id', 'parent_id']:
                feats[f"trace_{i}_{stats_func}"] = trace[i].agg(stats_func)

    else:
        feats['trace_length'] = -1

    # log特征
    if len(log) > 0:
        feats['log_length'] = len(log)
        log['message_length'] = log['message'].fillna("").map(len)
        log['log_info_length'] = log['message'].map(
            lambda x: x.split("INFO")).map(len)

    else:
        feats['log_length'] = -1

    # metric特征
    if len(metric) > 0:
        feats['metric_length'] = len(metric)
        feats['metric_value_timestamp_value_mean_std'] = metric.groupby(['timestamp'])[
            'value'].mean().std()
    else:
        feats['metric_length'] = -1

    return feats


def gen_label(train):
    col = np.zeros((train.shape[0], num_classes))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1

    return col


# 特征工程
all_ids = set([i.split("_")[0] for i in os.listdir("/data1/wyy/projects/_competition/thubdc/inputs/metric/")]) | \
    set([i.split("_")[0] for i in os.listdir("/data1/wyy/projects/_competition/thubdc/inputs/log/")]) | \
    set([i.split("_")[0] for i in os.listdir(
        "/data1/wyy/projects/_competition/thubdc/inputs/trace/")])
all_ids = list(all_ids)
print("IDs Length =", len(all_ids))
feature = pd.DataFrame(Parallel(n_jobs=64, backend="multiprocessing")(
    delayed(processing_feature)(f) for f in tqdm(all_ids)))

label = pd.read_csv(
    "/data1/wyy/projects/_competition/thubdc/labels/training_label.csv")
lb_encoder = LabelEncoder()
label['label'] = lb_encoder.fit_transform(label['source'])

# 特征处理
all_data = feature.merge(label[['id', 'label']].groupby(['id'], as_index=False)[
                         'label'].agg(list), how='left', on=['id']).set_index("id")

not_use = ['id', 'label']
feature_name = [i for i in all_data.columns if i not in not_use]
X = all_data[feature_name].replace([np.inf, -np.inf], 0).clip(-1e9, 1e9)
print(f"Feature Length = {len(feature_name)}")
print(f"Feature = {feature_name}")


# 训练过程
num_classes = 9
n_splits = 5
kf = MultilabelStratifiedKFold(
    n_splits=n_splits, random_state=3407, shuffle=True)
scaler = StandardScaler()
scaler_X = scaler.fit_transform(X.fillna(0).replace([np.inf, -np.inf], 0))

y = gen_label(all_data[all_data['label'].notnull()])
train_scaler_X = scaler_X[all_data['label'].notnull()]
test_scaler_X = scaler_X[all_data['label'].isnull()]

ovr_oof = np.zeros((len(train_scaler_X), num_classes))
ovr_preds = np.zeros((len(test_scaler_X), num_classes))


# 设置lgbm分类器的参数字典，作为函数传值传入
# best params：lr:0.05,max_depth:8,n_estimators:200
params = {
    # 'boosting_type': 'gbdt',  # 提升方法类型，可以是'gbdt'、'dart'、'goss'、'rf'
    'num_leaves': 40,  # 叶子节点数，控制模型的复杂度
    'max_depth': 10,  # 树的最大深度，-1表示不限制
    'learning_rate': 0.05,  # 学习率
    'n_estimators': 200,  # 弱学习器的数量
    'subsample_for_bin': 200000,  # 用于构建直方图的样本数
    'objective': None,  # 损失函数类型
    'class_weight': None,  # 类别权重，用于处理不平衡数据集
    'min_split_gain': 0.0,  # 分割阈值的最小增益
    'min_child_weight': 0.001,  # 叶子节点的最小样本权重和
    'min_child_samples': 30,  # 叶子节点的最小样本数
    'subsample': 0.75,  # 训练每棵树时使用的样本比例
    'subsample_freq': 0,  # 子样本的频率
    'colsample_bytree': 0.9,  # 构建每棵树时使用的特征比例（降低有利于减少模型的过拟合）
    'reg_alpha': 0.1,  # L1正则化项参数
    'reg_lambda': 0.0,  # L2正则化项参数
    'random_state': 0,  # 随机种子
    'n_jobs': 64,  # 并行工作的数量
    'silent': True,  # 是否打印运行信息
    'importance_type': 'split',  # 特征重要性计算方式，可以是'split'或'gain'
    'early_stopping_rounds': None,  # 提前停止的轮数
    'eval_metric': None  # 评估指标
}


# 循环交叉验证
for train_index, valid_index in kf.split(train_scaler_X, y):
    X_train, X_valid = train_scaler_X[train_index], train_scaler_X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]

    # LGBM模型参数优化
    clf = OneVsRestClassifier(lgb.LGBMClassifier(**params))
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
file_name = f"lgbm_{current_time}_{np.mean(score_metric['score'])}.csv"

# 保存为CSV文件
submit.to_csv("/data1/wyy/projects/_competition/thubdc/results/" +
              file_name, index=False)


# # 从StringIO对象中获取控制台输出内容
# output = output_buffer.getvalue()

# # # 将输出内容写入CSV文件
# with open("/data1/wyy/projects/_competition/thubdc/logs/lgbm_result.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Console Output"])
#     writer.writerow([output])
