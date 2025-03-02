import os
from moviepy.editor import VideoFileClip

# 设置输入和输出文件夹
input_dir = "/home/wangwc/project/CogVideoX/finetune/data/custom_video_dataset/videos/Lifting/video" 
output_dir = "/home/wangwc/project/CogVideoX/finetune/data/LiftingLifting/video"

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有文件
for filename in os.listdir(input_dir):
    if filename.endswith(".avi"):  # 只处理.avi文件
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".mp4"  # 替换扩展名为.mp4
        output_path = os.path.join(output_dir, output_filename)
        
        # 加载AVI文件并转换为MP4
        try:
            clip = VideoFileClip(input_path)
            clip.write_videofile(output_path, codec="libx264")
            print(f"转换成功: {input_path} -> {output_path}")
        except Exception as e:
            print(f"转换失败: {input_path} -> {output_path}, 错误: {e}")