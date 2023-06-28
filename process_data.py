import os
import shutil
import threading


def copy_files(source_dir, dest_dir):
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copyfile(source_path, dest_path)
            print(f"Copied {filename} to {dest_dir}")


# 设置源文件夹和目标文件夹的路径
source_folder = "data/testing"
dest_folder = "inputs"

# 创建目标文件夹中的子文件夹
os.makedirs(os.path.join(dest_folder, "log"), exist_ok=True)
os.makedirs(os.path.join(dest_folder, "trace"), exist_ok=True)
os.makedirs(os.path.join(dest_folder, "metric"), exist_ok=True)

# 创建线程列表
threads = []

# 复制 logs 文件夹下的文件
logs_thread = threading.Thread(target=copy_files, args=(
    os.path.join(source_folder, "log"), os.path.join(dest_folder, "log")))
threads.append(logs_thread)

# 复制 trace 文件夹下的文件
trace_thread = threading.Thread(target=copy_files, args=(
    os.path.join(source_folder, "trace"), os.path.join(dest_folder, "trace")))
threads.append(trace_thread)

# 复制 metric 文件夹下的文件
metric_thread = threading.Thread(target=copy_files, args=(
    os.path.join(source_folder, "metric"), os.path.join(dest_folder, "metric")))
threads.append(metric_thread)

# 启动所有线程
for thread in threads:
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

print("All files copied successfully.")
