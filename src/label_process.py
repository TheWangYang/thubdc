import pandas as pd

# 读取csv文件
df = pd.read_csv('thubdc_datasets/labels/training_label.csv', header=None)

# 统计每个标签的数量
label_counts = df.iloc[:, -1].value_counts()

# 计算每两个标签之间的数量比值
ratios = []
labels = label_counts.index.tolist()  # 获取所有标签列表
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        label1 = labels[i]
        label2 = labels[j]
        count1 = label_counts[label1]
        count2 = label_counts[label2]
        ratio = max(count1, count2) / min(count1, count2)
        ratios.append(ratio)

# 计算比值的众数和平均数
mode_ratio = max(set(ratios), key=ratios.count)
mean_ratio = sum(ratios) / len(ratios)

print("比值的众数:", mode_ratio)
print("比值的平均数:", mean_ratio)
