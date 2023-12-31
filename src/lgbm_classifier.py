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
# def processing_feature(file):
#     log, trace, metric, metric_df = pd.DataFrame(
#     ), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#     if os.path.exists(f"/data1/wyy/projects/_competition/thubdc/inputs/log/{file}_log.csv"):
#         log = pd.read_csv(
#             f"/data1/wyy/projects/_competition/thubdc/inputs/log/{file}_log.csv").sort_values(by=['timestamp']).reset_index(drop=True)

#     if os.path.exists(f"/data1/wyy/projects/_competition/thubdc/inputs/trace/{file}_trace.csv"):
#         trace = pd.read_csv(
#             f"/data1/wyy/projects/_competition/thubdc/inputs/trace/{file}_trace.csv").sort_values(by=['timestamp']).reset_index(drop=True)

#     if os.path.exists(f"/data1/wyy/projects/_competition/thubdc/inputs/metric/{file}_metric.csv"):
#         metric = pd.read_csv(
#             f"/data1/wyy/projects/_competition/thubdc/inputs/metric/{file}_metric.csv").sort_values(by=['timestamp']).reset_index(drop=True)

#     feats = {"id": file}

#     # trace特征
#     if len(trace) > 0:
#         feats['trace_length'] = len(trace)
#         feats[f"trace_status_code_std"] = trace['status_code'].apply("std")

#         # 创造timestamp的统计特征
#         # ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']
#         # ['mean', 'std', 'skew', 'nunique']

#         for stats_func in ['mean', 'std', 'skew', 'nunique']:
#             feats[f"trace_timestamp_{stats_func}"] = trace['timestamp'].apply(
#                 stats_func)

#         # 创造start_time的统计特征
#         for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
#             feats[f"trace_start_time_{stats_func}"] = trace['start_time'].apply(
#                 stats_func)

#         # 创造end_time的统计特征
#         for stats_func in ['count', 'nunique', 'sum', 'min', 'max', 'mean', 'std', 'sem', 'median', 'mad', 'var', 'skew']:
#             feats[f"trace_end_time_{stats_func}"] = trace['end_time'].apply(
#                 stats_func)

#         for stats_func in ['nunique']:
#             for i in ['host_ip', 'service_name', 'endpoint_name', 'trace_id', 'span_id', 'parent_id']:
#                 feats[f"trace_{i}_{stats_func}"] = trace[i].agg(stats_func)

#     else:
#         feats['trace_length'] = -1

#     # log特征
#     if len(log) > 0:
#         feats['log_length'] = len(log)
#         log['message_length'] = log['message'].fillna("").map(len)
#         log['log_info_length'] = log['message'].map(
#             lambda x: x.split("INFO")).map(len)

#     else:
#         feats['log_length'] = -1

#     # metric特征
#     if len(metric) > 0:
#         feats['metric_length'] = len(metric)
#         feats['metric_value_timestamp_value_mean_std'] = metric.groupby(['timestamp'])[
#             'value'].mean().std()
#     else:
#         feats['metric_length'] = -1

#     return feats


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
    col = np.zeros((train.shape[0], num_classes))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1

    return col


# 特征工程
all_ids = set([i.split("_")[0] for i in os.listdir("thubdc_datasets/inputs/metric/")]) | \
    set([i.split("_")[0] for i in os.listdir("thubdc_datasets/inputs/log/")]) | \
    set([i.split("_")[0] for i in os.listdir(
        "thubdc_datasets/inputs/trace/")])
all_ids = list(all_ids)
print("IDs Length =", len(all_ids))
feature = pd.DataFrame(Parallel(n_jobs=64, backend="multiprocessing")(
    delayed(processing_feature)(f) for f in tqdm(all_ids)))

label = pd.read_csv(
    "thubdc_datasets/labels/training_label.csv")
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
    'max_depth': 8,  # 树的最大深度，-1表示不限制
    'learning_rate': 0.05,  # 学习率
    'n_estimators': 200,  # 弱学习器的数量
    # 'subsample_for_bin': 200000,  # 用于构建直方图的样本数
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
    # 'early_stopping_rounds': None,  # 提前停止的轮数
    # 'eval_metric': None  # 评估指标
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
