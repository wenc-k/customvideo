import os
import cv2
import numpy as np

# 输入文件夹路径
images_dir = "images/"
masks_dir = "masks/"
output_dir = "masked_img/"

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 获取所有图片和mask文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

# 遍历图片并处理
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    mask_path = os.path.join(masks_dir, image_file.replace(".jpg", ".png"))
    
    # 检查是否存在对应的 mask 文件
    if os.path.exists(mask_path):
        # 读取图片和mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 检查图片和mask的尺寸是否一致
        if image.shape[:2] != mask.shape:
            print(f"Warning: Size mismatch between {image_file} and its mask.")
            continue

        # 应用mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        # 保存结果
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, masked_image)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"Mask not found for: {image_file}")
