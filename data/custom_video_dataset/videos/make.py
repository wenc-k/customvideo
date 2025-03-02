import os
import cv2

def get_video_info(video_path):
    """
    获取视频的宽度（W）、高度（H）和帧率（F）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数

    cap.release()
    return width, height, frame_count

def process_directory(directory):
    """
    处理目录下的所有 .avi 视频文件
    """
    if not os.path.isdir(directory):
        print(f"目录不存在: {directory}")
        return

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".avi"):
            video_path = os.path.join(directory, filename)
            width, height, fps = get_video_info(video_path)

            if width is not None and height is not None and fps is not None:
                print(f"视频文件: {filename}")
                print(f"宽度 (W): {width}, 高度 (H): {height}, 帧率 (F): {fps}")
                print("-" * 40)

if __name__ == "__main__":
    # 指定要处理的目录
    directory = "/home/wangwc/project/CogVideoX/finetune/data/custom_video_dataset/videos/Kicking-Front/video"  # 替换为你的目录路径
    process_directory(directory)