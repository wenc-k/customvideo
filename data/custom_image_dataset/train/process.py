import os
import json
from os.path import join, isdir, abspath, normpath, basename

def main():
    # 输入目录路径
    dir_name = '/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/train/part7' # TODO.
    dir_name = abspath(dir_name)
    
    # 读取motion_list.json
    # TODO.
    motion_base_path = '/home/wangwc/project/CogVideoX/finetune/data/custom_video_dataset/videos'
    motion_list = ["Run-Side", "Riding-Horse", "Kicking-Front", "Riding-Horse",
                   "PlayingCello", "Swing-SideAngle", "Lifting"]
    motion_list = [join(motion_base_path, motion) for motion in motion_list]
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(dir_name) if isdir(join(dir_name, d))]
    
    # 验证子目录数量与motion_list一致
    if len(subdirs) != len(motion_list):
        raise ValueError(f"子目录数量({len(subdirs)})与motion_list长度({len(motion_list)})不符")
    
    # 设置输出基目录（根据需求修改此处路径）
    output_base = '/home/wangwc/project/CogVideoX/finetune/exp_joint_result/part7/'  # TODO.
    os.makedirs(output_base, exist_ok=True)

    entries = []
    for subdir, motion_val in zip(subdirs, motion_list):
        subdir_path = join(dir_name, subdir)
        
        # 实例数据根路径
        instance_data_root_id = abspath(subdir_path)
        
        # 实例数据运动路径（直接使用列表中的值）
        instance_data_root_motion = motion_val
        
        # 构建output_dir：输出基目录 + 子目录名_运动路径结尾名
        motion_end = basename(normpath(motion_val.strip('/')))  # 处理路径结尾
        output_subdir = f"{subdir}_{motion_end}"
        output_dir = abspath(join(output_base, output_subdir))
        
        # 查找参考图像路径
        images_dir = join(subdir_path, 'images')
        if not isdir(images_dir):
            raise FileNotFoundError(f"{images_dir}不存在")
        allowed_exts = ('.jpg', '.png', '.jpeg')
        matching_files = []
        for file in os.listdir(images_dir):
            if file.startswith('00') and file.lower().endswith(allowed_exts) or file.startswith('0') and file.lower().endswith(allowed_exts):
                matching_files.append(file)
        if not matching_files:
            raise FileNotFoundError(f"{images_dir}中未找到以00开头的图片")
        matching_files.sort()  # 按文件名排序取第一个
        ref_image_path = abspath(join(images_dir, matching_files[0]))
        
        # 添加到条目
        entries.append({
            "instance_data_root_id": instance_data_root_id,
            "instance_data_root_motion": instance_data_root_motion,
            "output_dir": output_dir,
            "ref_image_path": ref_image_path
        })
    
    # 写入train.json
    output_json = join(dir_name, 'train.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)
    print(f"成功生成 {output_json}，包含 {len(entries)} 个条目")

if __name__ == "__main__":
    main()