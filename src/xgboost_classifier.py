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
    if os.path.exists(f"thubdc_datasets/inputs/log/{file}_log.csv"):
        log = pd.read_csv(
            f"thubdc_datasets/inputs/log/{file}_log.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"thubdc_datasets/inputs/trace/{file}_trace.csv"):
        trace = pd.read_csv(
            f"thubdc_datasets/inputs/trace/{file}_trace.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"thubdc_datasets/inputs/metric/{file}_metric.csv"):
        metric = pd.read_csv(
            f"thubdc_datasets/inputs/metric/{file}_metric.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    feats = {"id": file}
    # 做trace的特征
    if len(trace) > 0:
        feats['trace_length'] = len(trace)
        feats[f"trace_status_code_std"] = trace['status_code'].apply("std")

        # -----------------------------------------------创造timestamp的统计特征---------------------------------------------------
        # ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
        # ['mean', 'std', 'skew', 'nunique']:
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
            feats[f"trace_timestamp_{stats_func}"] = trace['timestamp'].apply(
                stats_func)

        # # 计算百分位数
        # feats["trace_timestamp_75th_percentile"] = np.percentile(
        #     trace['timestamp'], 75)

        # # 计算众数
        # feats["trace_timestamp_mode"] = trace['timestamp'].mode()

        # # 计算峰度
        # feats["trace_timestamp_kurtosis"] = trace['timestamp'].kurtosis()

        # # 计算缺失值数量
        # feats["trace_timestamp_missing_count"] = trace['timestamp'].isnull().sum()

        # # 进行相关性分析（示例，计算'timestamp'列与其他列的相关性）
        # correlation = trace.corr()['timestamp']
        # feats["trace_timestamp_correlation"] = correlation.drop('timestamp')

        # ---------------------------------------------------创造start_time的统计特征-----------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
            feats[f"trace_start_time_{stats_func}"] = trace['start_time'].apply(
                stats_func)

        # ------------------------------------------------------创造end_time的统计特征----------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
            feats[f"trace_end_time_{stats_func}"] = trace['end_time'].apply(
                stats_func)

        # ------------------------------------------------创造start_time和end_time的统计特征--------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
            feats[f"trace_time_jia_{stats_func}"] = (
                trace['end_time'] + trace['start_time']).apply(stats_func)

            feats[f"trace_time_jian_{stats_func}"] = (
                trace['end_time'] - trace['start_time']).apply(stats_func)

            feats[f"trace_time_cheng_{stats_func}"] = (
                trace['end_time'] * trace['start_time']).apply(stats_func)

            feats[f"trace_time_chu_{stats_func}"] = (
                trace['end_time'] / trace['start_time']).apply(stats_func)

            # 平方根特征
            feats[f"trace_time_jia_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_time_jia_{stats_func}"])
            feats[f"trace_time_jian_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_time_jian_{stats_func}"])
            feats[f"trace_time_cheng_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_time_cheng_{stats_func}"])
            feats[f"trace_time_chu_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_time_chu_{stats_func}"])

            # 立方根特征
            feats[f"trace_time_jia_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_time_jia_{stats_func}"])
            feats[f"trace_time_jian_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_time_jian_{stats_func}"])
            feats[f"trace_time_cheng_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_time_cheng_{stats_func}"])
            feats[f"trace_time_chu_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_time_chu_{stats_func}"])

        # --------------------------------------------------创造其他非数字列的标准差-------------------------------------------------
        for stats_func in ['nunique']:
            for i in ['host_ip', 'service_name', 'endpoint_name', 'trace_id', 'span_id', 'parent_id']:
                feats[f"trace_{i}_{stats_func}"] = trace[i].agg(stats_func)

        # ------------------------------------------创造service_name和endpoint_name之间的编码特征------------------------------------
        from sklearn.preprocessing import LabelEncoder

        # 创建LabelEncoder对象
        label_encoder = LabelEncoder()

        # 填充空值
        trace['service_name'].fillna('missing', inplace=True)
        trace['endpoint_name'].fillna('missing', inplace=True)

        # 对service_name和endpoint_name列进行Label编码
        trace['service_name_encoded'] = label_encoder.fit_transform(
            trace['service_name'])
        trace['endpoint_name_encoded'] = label_encoder.fit_transform(
            trace['endpoint_name'])

        # 创建编码特征
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
            feats[f'trace_service_endpoint_jia_{stats_func}'] = (
                trace['service_name_encoded'] + trace['endpoint_name_encoded']).apply(stats_func)

            feats[f'trace_service_endpoint_jian_{stats_func}'] = (
                trace['service_name_encoded'] - trace['endpoint_name_encoded']).apply(stats_func)

            feats[f'trace_service_endpoint_cheng_{stats_func}'] = (
                trace['service_name_encoded'] * trace['endpoint_name_encoded']).apply(stats_func)

            feats[f'trace_service_endpoint_chu_{stats_func}'] = (
                trace['service_name_encoded'] / trace['endpoint_name_encoded']).apply(stats_func)

            # 平方根特征
            feats[f"trace_service_endpoint_jia_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_service_endpoint_jia_{stats_func}"])
            feats[f"trace_service_endpoint_jian_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_service_endpoint_jian_{stats_func}"])
            feats[f"trace_service_endpoint_cheng_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_service_endpoint_cheng_{stats_func}"])
            feats[f"trace_service_endpoint_chu_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_service_endpoint_chu_{stats_func}"])

            # 立方根特征
            feats[f"trace_service_endpoint_jia_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_service_endpoint_jia_{stats_func}"])
            feats[f"trace_service_endpoint_jian_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_service_endpoint_jian_{stats_func}"])
            feats[f"trace_service_endpoint_cheng_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_service_endpoint_cheng_{stats_func}"])
            feats[f"trace_service_endpoint_chu_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_service_endpoint_chu_{stats_func}"])

    else:
        feats['trace_length'] = -1

    # 做log的特征
    if len(log) > 0:
        feats['log_length'] = len(log)
        log['message_length'] = log['message'].fillna("").map(len)
        log['log_info_length'] = log['message'].map(
            lambda x: x.split("INFO")).map(len)

    else:
        feats['log_length'] = -1

    # 做metric的特征
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
all_ids = set([i.split("_")[0] for i in os.listdir("thubdc_datasets/inputs/metric/")]) |\
    set([i.split("_")[0] for i in os.listdir("thubdc_datasets/inputs/log/")]) |\
    set([i.split("_")[0] for i in os.listdir(
        "thubdc_datasets/inputs/trace/")])

all_ids = list(all_ids)
print("IDs Length =", len(all_ids))

# 从所有的log/trace/metric文件夹中提取特征
feature = pd.DataFrame(Parallel(n_jobs=64, backend="multiprocessing")(
    delayed(processing_feature)(f) for f in tqdm(all_ids)))


# 获得所有训练标签
label = pd.read_csv(
    "thubdc_datasets/labels/training_label.csv")
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

print("all_data.dtypes:", all_data.dtypes)

# 从all_data中提取特征列，并对其中的无穷大值和负无穷大值进行替换和剪裁操作
X = all_data[feature_name].replace([np.inf, -np.inf], 0).clip(-1e9, 1e9)

# print("X.dtypes:", X.dtypes)
input()

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


# 设置lgbm分类器的参数字典，作为函数传值传入
# best params：lr:0.05,max_depth:8,n_estimators:200
params = {
    'max_depth': 8,  # 树的最大深度，-1表示不限制
    'learning_rate': 0.05,  # 学习率
    'n_estimators': 200,  # 弱学习器的数量
    # 'boosting_type': 'gbdt',  # 提升方法类型，可以是'gbdt'、'dart'、'goss'、'rf'
    # 'num_leaves': 40,  # 叶子节点数，控制模型的复杂度
    # 'objective': None,  # 损失函数类型
    # 'class_weight': None,  # 类别权重，用于处理不平衡数据集
    # 'min_split_gain': 0.0,  # 分割阈值的最小增益
    # 'min_child_weight': 0.002,  # 叶子节点的最小样本权重和
    # 'min_child_samples': 30,  # 叶子节点的最小样本数
    'subsample': 0.8,  # 训练每棵树时使用的样本比例
    # 'subsample_for_bin': 200000,  # 用于构建直方图的样本数
    # 'subsample_freq': 0,  # 子样本的频率
    # 'colsample_bytree': 0.9,  # 构建每棵树时使用的特征比例（降低有利于减少模型的过拟合）
    'reg_alpha': 0.5,  # L1正则化项参数
    'reg_lambda': 0.5,  # L2正则化项参数
    # 'gamma': 3.0,
    'random_state': 0,  # 随机种子
    'n_jobs': 64,  # 并行工作的数量
    # 'silent': True,  # 是否打印运行信息
    # 'importance_type': 'split',  # 特征重要性计算方式，可以是'split'或'gain'
    # 'early_stopping_rounds': None,  # 提前停止的轮数
    # 'eval_metric': None  # 评估指标
}


# 开始训练过程
for train_index, valid_index in kf.split(train_scaler_X, y):
    X_train, X_valid = train_scaler_X[train_index], train_scaler_X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    # 加入多个参数
    clf = OneVsRestClassifier(XGBClassifier(**params))
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
file_name = f"xgboost_{current_time}_{np.mean(score_metric['score'])}.csv"

# 保存为CSV文件
submit.to_csv("/data1/wyy/projects/_competition/thubdc/results/" +
              file_name, index=False)
