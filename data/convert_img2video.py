import os
import cv2

from PIL import Image
import numpy as np
import re

def crop_and_resize_pil(image_pil, target_size=(720, 480), crop_position='center'):
    """
    使用PIL实现的裁剪和缩放函数
    target_size格式: (width, height)
    """
    orig_width, orig_height = image_pil.size
    target_width, target_height = target_size
    
    orig_ratio = orig_width / orig_height
    target_ratio = target_width / target_height

    if orig_ratio > target_ratio:
        crop_height = orig_height
        crop_width = int(target_ratio * crop_height)
    else:
        crop_width = orig_width
        crop_height = int(crop_width / target_ratio)

    left = (orig_width - crop_width) // 2
    top = (orig_height - crop_height) // 2
    
    if crop_position == 'top':
        top = 0
    elif crop_position == 'bottom':
        top = orig_height - crop_height
    elif crop_position == 'left':
        left = 0
    elif crop_position == 'right':
        left = orig_width - crop_width

    cropped = image_pil.crop((left, top, left + crop_width, top + crop_height))
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    return resized

process_dir_path = '/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/train/part4/'

# 目标尺寸 (width, height)
TARGET_SIZE = (720, 480)

for entry in os.listdir(process_dir_path):
    full_path = os.path.join(process_dir_path, entry)
    dir_name = os.path.basename(full_path)
    # 生成prompts.txt 保存文件名到对应video数目的prompt
    processed = dir_name.replace('_', ' ')
    processed = re.sub(r'\d+', '', processed)
    processed = f"{processed.strip()} *"


    images_dir = os.path.join(full_path, 'images')
    output_dir = os.path.join(full_path, 'videos')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    # 创建videos.txt文件
    video_file = os.path.join(full_path, 'videos.txt')
    # 创建prompts文件
    prompt_file = os.path.join(full_path, 'prompts.txt')
    # 获取所有img文件
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPG"))]

    # 视频参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1  # 单帧视频
    # 生成单张视频并且导入到videos/; 生成crop图片导入到images/；删除原先所有图片
    for image_file in image_files:
        # 路径处理
        image_path = os.path.join(images_dir, image_file)
        output_path = os.path.join(output_dir, image_file.replace(".jpg", ".mp4").replace(".png", ".mp4").replace(".jpeg", ".mp4").replace(".JPG", ".mp4"))
        try:
            # 使用PIL读取并处理
            pil_image = Image.open(image_path)
            processed_image = crop_and_resize_pil(pil_image, TARGET_SIZE)

            processed_image.save(image_path)
            print(f"Saved cropped image: {image_path}")
            
            # 转换为OpenCV格式 (BGR)
            cv_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
            
            # 创建视频写入对象（注意尺寸参数顺序是 (width, height)）
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, TARGET_SIZE)
            
            if video_writer.isOpened():
                video_writer.write(cv_image)
                print(f"Success: {image_file} -> {output_path}")
            else:
                print(f"Failed to initialize video writer for {image_file}")

            video_name = os.path.basename(output_path)
            video_path = 'videos/' + video_name
            with open(video_file, 'a+', encoding='utf-8') as f:
                f.write(video_path+'\n')
            with open(prompt_file, 'a+', encoding='utf-8') as f:
                f.write(processed+'\n')
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
        finally:
            if 'video_writer' in locals():
                video_writer.release()


print("Processing completed!")