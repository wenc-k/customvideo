import os
import cv2

# 输入图片目录
images_dir = "masked_img/"
output_dir = "masked_videos/"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取目录中所有jpg文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

# 视频编码器和帧率
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
fps = 1

for image_file in image_files:
    # 读取图片路径
    image_path = os.path.join(images_dir, image_file)
    output_path = os.path.join(output_dir, image_file.replace(".jpg", ".mp4"))
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_file}")
        continue

    # 获取图片尺寸
    height, width, _ = image.shape

    # 初始化视频写入对象
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入单帧
    video_writer.write(image)

    # 释放视频写入对象
    video_writer.release()
    print(f"Processed and saved video: {output_path}")
