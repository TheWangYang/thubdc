from datetime import datetime
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
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
    
    
    # 除了大多数选手提到诸如endtime-starttime、start_time diff、timestamp diff，会从id和service、host、endpoint进行交叉统计，
    # 同时构建了host、service、endpoint统计结果与全局id统计结果的相关特征，比如host_timestamp_max/timestamp_max。
    # 做trace的特征
    if len(trace) > 0:
        feats['trace_length'] = len(trace)
        feats['trace_status_code_std'] = trace['status_code'].apply('std')
        
        # # ------------------------------------------------------创造status_code对应的高级特征---------------------------------------------------
        # # 按照状态码进行分组
        # trace_status_code_failure_data = trace.loc[trace['status_code'] != 200]
        # # 进一步对请求失败的数据进行处理和构建特征
        # for stats_func in ['count', 'mean']:
        #     feats[f'trace_status_code_failure_data_{stats_func}'] = trace_status_code_failure_data['status_code'].apply(stats_func)

        
        # ----------------------------------------------按照host_ip和service_name分组，做timestamp的diff特征------------------------------------
        trace_grouped = trace.groupby(['host_ip', 'service_name'])
        trace['trace_timestamp_diff'] = trace_grouped['timestamp'].diff()
        trace['trace_start_time_diff'] = trace_grouped['start_time'].diff()

        # 对 diff 列进行一些统计特征，例如平均值、标准差等
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f"trace_timestamp_diff_{stats_func}"] = trace['trace_timestamp_diff'].apply(
                stats_func)

        # # ---------------------------------------------------创造timestamp和start_time的相互关系特征----------------------------------
        # for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
        #     feats[f"trace_timestamp_start_time_jia_{stats_func}"] = (
        #         trace['timestamp'] + trace['start_time']).apply(stats_func)

        #     feats[f"trace_timestamp_start_time_jian_{stats_func}"] = (
        #         trace['timestamp'] - trace['start_time']).apply(stats_func)

        #     feats[f"trace_timestamp_start_time_cheng_{stats_func}"] = (
        #         trace['timestamp'] * trace['start_time']).apply(stats_func)

        #     feats[f"trace_timestamp_start_time_chu_{stats_func}"] = (
        #         trace['timestamp'] / trace['start_time']).apply(stats_func)

        # # ---------------------------------------------------创造timestamp的diff特征与start_time交互的特征------------------------------
        # for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
        #     feats[f"trace_timestamp_diff_{stats_func}_start_time_jia_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] + trace['start_time']).apply(stats_func)

        #     feats[f"trace_timestamp_diff_{stats_func}_start_time_jian_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] - trace['start_time']).apply(stats_func)

        #     feats[f"trace_timestamp_diff_{stats_func}_start_time_cheng_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] * trace['start_time']).apply(stats_func)

        #     feats[f"trace_timestamp_diff_{stats_func}_start_time_chu_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] / trace['start_time']).apply(stats_func)

        # ---------------------------------------------------创造timestamp与end_time之间的特征关系--------------------------------------
        # for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
        #     feats[f"trace_timestamp_end_time_jia_{stats_func}"] = (
        #         trace['timestamp'] + trace['end_time']).apply(stats_func)

        #     feats[f"trace_timestamp_end_time_jian_{stats_func}"] = (
        #         trace['timestamp'] - trace['end_time']).apply(stats_func)

        #     feats[f"trace_timestamp_end_time_cheng_{stats_func}"] = (
        #         trace['timestamp'] * trace['end_time']).apply(stats_func)

        #     feats[f"trace_timestamp_end_time_chu_{stats_func}"] = (
        #         trace['timestamp'] / trace['end_time']).apply(stats_func)

        # # ---------------------------------------------------创造timestamp的diff特征与end_time交互的特征------------------------------
        # for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
        #     feats[f"trace_timestamp_diff_{stats_func}_end_time_jia_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] + trace['end_time']).apply(stats_func)

        #     feats[f"trace_timestamp_diff_{stats_func}_end_time_jian_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] - trace['end_time']).apply(stats_func)

        #     feats[f"trace_timestamp_diff_{stats_func}_end_time_cheng_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] * trace['end_time']).apply(stats_func)

        #     feats[f"trace_timestamp_diff_{stats_func}_end_time_chu_{stats_func}"] = (
        #         trace[f"trace_timestamp_diff_{stats_func}"] / trace['end_time']).apply(stats_func)

        # --------------------------------------------------创造timestamp的统计特征---------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f"trace_timestamp_{stats_func}"] = trace['timestamp'].apply(
                stats_func)

            # 创建timestamp的平方根特征
            feats[f"trace_timestamp_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_timestamp_{stats_func}"])

            # 创建timestamp的立方根特征
            feats[f"trace_timestamp_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_timestamp_{stats_func}"])

        # ---------------------------------------------------创造start_time的统计特征-----------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f"trace_start_time_{stats_func}"] = trace['start_time'].apply(
                stats_func)

            # 创建start_time的平方根特征
            feats[f"trace_start_time_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_start_time_{stats_func}"])

            # 创建start_time的立方根特征
            feats[f"trace_start_time_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_start_time_{stats_func}"])

        # --------------------------------------创造timestamp的diff特征与start_time的统计特征的交互特征-----------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            for inner_stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
                feats[f"trace_timestamp_diff_{stats_func}_start_time_status_jia_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] + feats[f"trace_start_time_{inner_stats_func}"]

                feats[f"trace_timestamp_diff_{stats_func}_start_time_status_jian_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] - feats[f"trace_start_time_{inner_stats_func}"]

                feats[f"trace_timestamp_diff_{stats_func}_start_time_status_cheng_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] * feats[f"trace_start_time_{inner_stats_func}"]

                feats[f"trace_timestamp_diff_{stats_func}_start_time_status_chu_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] / feats[f"trace_start_time_{inner_stats_func}"]

        # -----------------------------------------------------创造end_time的统计特征----------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f"trace_end_time_{stats_func}"] = trace['end_time'].apply(
                stats_func)

            # 创建end_time的平方根特征
            feats[f"trace_end_time_sqrt_{stats_func}"] = np.sqrt(
                feats[f"trace_end_time_{stats_func}"])

            # 创建end_time的立方根特征
            feats[f"trace_end_time_cbrt_{stats_func}"] = np.cbrt(
                feats[f"trace_end_time_{stats_func}"])

        # ------------------------------------------创造timestamp的diff特征与end_time的统计特征的交互特征------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            for inner_stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
                feats[f"trace_timestamp_diff_{stats_func}_end_time_status_jia_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] + feats[f"trace_end_time_{inner_stats_func}"]

                feats[f"trace_timestamp_diff_{stats_func}_end_time_status_jian_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] - feats[f"trace_end_time_{inner_stats_func}"]

                feats[f"trace_timestamp_diff_{stats_func}_end_time_status_cheng_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] * feats[f"trace_end_time_{inner_stats_func}"]

                feats[f"trace_timestamp_diff_{stats_func}_end_time_status_chu_{inner_stats_func}"] = feats[
                    f"trace_timestamp_diff_{stats_func}"] / feats[f"trace_end_time_{inner_stats_func}"]

        # ------------------------------------------------创造end_time和start_time的统计特征--------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f"trace_time_jia_{stats_func}"] = (
                trace['end_time'] + trace['start_time']).apply(stats_func)

            feats[f"trace_time_jian_{stats_func}"] = (
                trace['end_time'] - trace['start_time']).apply(stats_func)

            feats[f"trace_time_cheng_{stats_func}"] = (
                trace['end_time'] * trace['start_time']).apply(stats_func)

            feats[f"trace_time_chu_{stats_func}"] = (
                trace['end_time'] / trace['start_time']).apply(stats_func)

        # --------------------------------------------------创造其他非数字列的特征-------------------------------------------------
        for stats_func in ['nunique']:
            for i in ['host_ip', 'service_name', 'endpoint_name', 'trace_id', 'span_id', 'parent_id']:
                feats[f"trace_{i}_{stats_func}"] = trace[i].agg(stats_func)

        # ------------------------------------------创造service_name和endpoint_name之间的编码特征------------------------------------
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
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f'trace_service_endpoint_jia_{stats_func}'] = (
                trace['service_name_encoded'] + trace['endpoint_name_encoded']).apply(stats_func)

            feats[f'trace_service_endpoint_jian_{stats_func}'] = (
                trace['service_name_encoded'] - trace['endpoint_name_encoded']).apply(stats_func)

            feats[f'trace_service_endpoint_cheng_{stats_func}'] = (
                trace['service_name_encoded'] * trace['endpoint_name_encoded']).apply(stats_func)

            feats[f'trace_service_endpoint_chu_{stats_func}'] = (
                trace['service_name_encoded'] / trace['endpoint_name_encoded']).apply(stats_func)
            
            # 创造trace_timestamp_diff与service_name_encoded之间的交互关系
            feats[f'trace_timestamp_diff_service_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] + trace['service_name_encoded']).apply(stats_func)
            feats[f'trace_timestamp_diff_service_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] - trace['service_name_encoded']).apply(stats_func)
            feats[f'trace_timestamp_diff_service_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] * trace['service_name_encoded']).apply(stats_func)
            feats[f'trace_timestamp_diff_service_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] / trace['service_name_encoded']).apply(stats_func)
            
            
            # 创造trace_timestamp_diff与endpoint_name_encoded之间的交互关系
            feats[f'trace_timestamp_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] + trace['endpoint_name_encoded']).apply(stats_func)
            feats[f'trace_timestamp_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] - trace['endpoint_name_encoded']).apply(stats_func)
            feats[f'trace_timestamp_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] * trace['endpoint_name_encoded']).apply(stats_func)
            feats[f'trace_timestamp_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_timestamp_diff'] / trace['endpoint_name_encoded']).apply(stats_func)
            
            
            # 创造trace_start_time_diff与service_name_encoded之间的交互关系
            feats[f'trace_start_time_diff_service_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] + trace['service_name_encoded']).apply(stats_func)
            feats[f'trace_start_time_diff_service_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] - trace['service_name_encoded']).apply(stats_func)
            feats[f'trace_start_time_diff_service_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] * trace['service_name_encoded']).apply(stats_func)
            feats[f'trace_start_time_diff_service_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] / trace['service_name_encoded']).apply(stats_func)
            
            
            # 创造trace_start_time_diff与endpoint_name_encoded之间的交互关系
            feats[f'trace_start_time_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] + trace['endpoint_name_encoded']).apply(stats_func)
            feats[f'trace_start_time_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] - trace['endpoint_name_encoded']).apply(stats_func)
            feats[f'trace_start_time_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] * trace['endpoint_name_encoded']).apply(stats_func)
            feats[f'trace_start_time_diff_endpoint_name_encoded_{stats_func}'] = (trace['trace_start_time_diff'] / trace['endpoint_name_encoded']).apply(stats_func)

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
        # 参考周周星做特征
        feats['log_length'] = len(log)
        feats['log_service_nunique'] = log['service'].nunique()
        feats['message_length_std'] = log['message'].fillna("").map(len).std()
        feats['message_length_ptp'] = log['message'].fillna(
            "").map(len).agg('ptp')
        feats['log_info_length'] = log['message'].map(
            lambda x: x.split("INFO")).map(len).agg('ptp')

        # developed feature
        # text_list = ['异常','错误','error','user','mysql','true','失败']
        text_list = ['user', 'mysql']
        for text in text_list:
            # feats[f'message_{text}_sum'] = log['message'].str.contains(text, case=False).sum()
            feats[f'message_{text}_mean'] = log['message'].str.contains(
                text, case=False).mean()
        feats[f'message_mysql_mean'] = 1 if feats[f'message_mysql_mean'] > 0 else 0

        # ---------------------------------------------创建log中的timestamp的特征--------------------------------------------------
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f"log_timestamp_{stats_func}"] = log['timestamp'].apply(
                stats_func)

            # 创建timestamp的平方根特征
            feats[f"log_timestamp_sqrt_{stats_func}"] = np.sqrt(
                feats[f"log_timestamp_{stats_func}"])

            # 创建timestamp的立方根特征
            feats[f"log_timestamp_cbrt_{stats_func}"] = np.cbrt(
                feats[f"log_timestamp_{stats_func}"])

        # ---------------------------------------------创造log中message的编码特征------------------------------------
        # 创建LabelEncoder对象
        label_encoder = LabelEncoder()

        log['message'].fillna('missing', inplace=True)

        # 对message列进行Label编码
        log['log_message_encoded'] = label_encoder.fit_transform(
            log['message'])

        # 创建编码特征
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f'log_message_encoded_{stats_func}'] = log['log_message_encoded'].apply(
                stats_func)

            # 平方根特征
            feats[f"log_message_encoded_sqrt_{stats_func}"] = np.sqrt(
                feats[f"log_message_encoded_{stats_func}"])

            # 立方根特征
            feats[f"log_message_encoded_cbrt_{stats_func}"] = np.cbrt(
                feats[f"log_message_encoded_{stats_func}"])

        # --------------------------------------------------创造log中service的编码特征-----------------------------------------
        # 创建LabelEncoder对象
        label_encoder = LabelEncoder()

        log['service'].fillna('missing', inplace=True)

        # 对message列进行Label编码
        log['log_service_encoded'] = label_encoder.fit_transform(
            log['service'])

        # 创建编码特征
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f'log_service_encoded_{stats_func}'] = log['log_service_encoded'].apply(
                stats_func)

            # 平方根特征
            feats[f"log_service_encoded_sqrt_{stats_func}"] = np.sqrt(
                feats[f"log_service_encoded_{stats_func}"])

            # 立方根特征
            feats[f"log_service_encoded_cbrt_{stats_func}"] = np.cbrt(
                feats[f"log_service_encoded_{stats_func}"])

        # -------------------------------------------创造log中service和message之间的交互特征---------------------------------
        # 创建编码特征
        for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew', 'ptp']:
            feats[f'log_message_service_encoded_jia_{stats_func}'] = (
                log['log_message_encoded'] + log['log_service_encoded']).apply(stats_func)

            feats[f'log_message_service_encoded_jian_{stats_func}'] = (
                log['log_message_encoded'] - log['log_service_encoded']).apply(stats_func)

            feats[f'log_message_service_encoded_cheng_{stats_func}'] = (
                log['log_message_encoded'] * log['log_service_encoded']).apply(stats_func)

            feats[f'log_message_service_encoded_chu_{stats_func}'] = (
                log['log_message_encoded'] / log['log_service_encoded']).apply(stats_func)

    else:
        feats['log_length'] = -1

    # # 做metric的特征
    # if len(metric) > 0:
    #     feats['metric_length'] = len(metric)
    #     feats['metric_value_timestamp_value_mean_std'] = metric.groupby(['timestamp'])[
    #         'value'].mean().std()

    #     # ----------------------------------------------创建metric中的tags编码特征--------------------------------------
    #     # 创建LabelEncoder对象
    #     label_encoder = LabelEncoder()

    #     metric['tags'].fillna('missing', inplace=True)

    #     # 对message列进行Label编码
    #     metric['metric_tags_encoded'] = label_encoder.fit_transform(
    #         metric['tags'])

    #     for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
    #         feats[f'metric_tags_encoded_{stats_func}'] = metric['metric_tags_encoded'].apply(
    #             stats_func)

    #         # 平方根特征
    #         feats[f"metric_tags_encoded_sqrt_{stats_func}"] = np.sqrt(
    #             feats[f"metric_tags_encoded_{stats_func}"])

    #         # 立方根特征
    #         feats[f"metric_tags_encoded_cbrt_{stats_func}"] = np.cbrt(
    #             feats[f"metric_tags_encoded_{stats_func}"])

    # else:
    #     feats['metric_length'] = -1

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
feature = pd.DataFrame(Parallel(n_jobs=-1, backend="multiprocessing")(
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

# feats['trace_host_timestamp_max'] = trace.groupby(['host_ip'])['timestamp'].agg('max')
# feats['trace_service_timestamp_max'] = trace.groupby(['service_name'])['timestamp'].agg('max')
# feats['trace_endpoint_name_timestamp_max'] = trace.groupby(['endpoint_name'])['timestamp'].agg('max')
# feats['trace_timestamp_max'] = trace_grouped['timestamp'].agg('max')

# 设置一个特征值过滤函数来得到需要的列
def correlation(data, threshold):
    col_corr = []
    corr_matrix = data.corr()
    for i in range(len(corr_matrix)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.append(colname)
    return list(set(col_corr))

# 设置不使用的列名
used_cols = correlation(all_data, 0.98)

print("used_cols: ", used_cols)

# 将其他使用的列名都作为特征
feature_name = [i for i in all_data.columns if i in used_cols]

print("all_data.dtypes:", all_data.dtypes)

# 从all_data中提取特征列，并对其中的无穷大值和负无穷大值进行替换和剪裁操作
X = all_data[feature_name].replace([np.inf, -np.inf], 0).clip(-1e9, 1e9)

# print("X.dtypes:", X.dtypes)

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



# 设置catboost分类器的参数
params = {
    'iterations': 1000,  # 迭代次数
    'learning_rate': 0.1,  # 学习率
    'depth': 6,  # 决策树深度
    'l2_leaf_reg': 3,  # L2 正则化参数
    'border_count': 32,  # 训练时特征值的取值范围
    'thread_count': -1,  # 线程数，根据需要调整，设置为最大
    'random_seed': 42,  # 随机种子，可选
    'model_size_reg': 0,  # 模型大小正则化参数
    'rsm':1,
    'loss_function': 'Logloss',
    # 'border_count':,
    #'feature_border_type':,
    # 'use_best_model': True,
    'task_type': 'GPU',
    'devices': [0,1,2,3],
}


# 使用手动调参的训练过程
for train_index, valid_index in kf.split(train_scaler_X, y):
    X_train, X_valid = train_scaler_X[train_index], train_scaler_X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    # 加入多个参数
    clf = OneVsRestClassifier(CatBoostClassifier(**params))
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
file_name = f"catboost_{current_time}_{np.mean(score_metric['score'])}.csv"

# 保存为CSV文件
submit.to_csv("/data1/wyy/projects/_competition/thubdc/results/" +
              file_name, index=False)
