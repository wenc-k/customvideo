import os
import cv2

# 目标目录
root_dir = "/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/train/part6"

# 获取主目录的直接子目录
for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)
    if os.path.isdir(dir_path):  # 确保是目录
        # 在子目录中创建空文件
        file_path = os.path.join(dir_path, "eval_pro_50.txt")
        try:
            with open(file_path, "w") as f:
                pass  # 创建空文件
            print(f"已创建文件: {file_path}")
        except Exception as e:
            print(f"创建文件 {file_path} 时发生错误: {e}")