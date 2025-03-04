import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import clip
import cv2
from transformers import AutoImageProcessor, AutoModel

import open_clip

from easydict import EasyDict as edict
import glob
from third_party.RAFT.core.raft import RAFT
from third_party.RAFT.core.utils_core.utils import InputPadder

from omegaconf import OmegaConf
from third_party.amt.utils.build_utils import build_from_cfg
from third_party.amt.utils.utils import InputPadder
from third_party.amt.utils.utils import (
    img2tensor, tensor2img,
    check_dim_and_resize
    )

import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import font_manager
import csv


class DynamicDegree:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()
    

    def load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()


    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()


    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}


    def infer(self, video_path):
        with torch.no_grad():
            if video_path.endswith('.mp4'):
                frames = self.get_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.get_frames_from_img_folder(video_path)
            else:
                raise NotImplementedError
            self.set_params(frame=frames[0], count=len(frames))
            static_score = []
            for image1, image2 in zip(frames[:-1], frames[1:]):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)
            return np.mean(static_score)


    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False


    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # get fps
        interval = max(1, round(fps / 8))
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        frame_list = self.extract_frame(frame_list, interval)
        return frame_list 
    
    
    def extract_frame(self, frame_list, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract

    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
        'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
        'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list

class FrameProcess:
    def __init__(self):
        pass


    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        return frame_list 
    

    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
                'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
                'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list


    def extract_frame(self, frame_list, start_from=0):
        extract = []
        for i in range(start_from, len(frame_list), 2):
            extract.append(frame_list[i])
        return extract
    
class MotionSmoothness:
    def __init__(self, config, ckpt, device):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.initialization()
        self.load_model()

    
    def load_model(self):
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        print(f'Loading [{network_name}] from [{ckpt_path}]...')
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()


    def initialization(self):
        if self.device == 'cuda':
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192*8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        self.fp = FrameProcess()


    def motion_score(self, video_path):
        iters = int(self.niters)
        # get inputs
        if video_path.endswith('.mp4'):
            frames = self.fp.get_frames(video_path)
        elif os.path.isdir(video_path):
            frames = self.fp.get_frames_from_img_folder(video_path)
        else:
            raise NotImplementedError
        frame_list = self.fp.extract_frame(frames, start_from=0)
        # print(f'Loading [images] from [{video_path}], the number of images = [{len(frame_list)}]')
        inputs = [img2tensor(frame).to(self.device) for frame in frame_list]
        assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"
        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        if scale < 1:
            print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        # -----------------------  Interpolater ----------------------- 
        # print(f'Start frame interpolation:')
        for i in range(iters):
            # print(f'Iter {i+1}. input_frames={len(inputs)} output_frames={2*len(inputs)-1}')
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        # -----------------------  cal_vfi_score ----------------------- 
        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(out) for out in outputs]
        vfi_score = self.vfi_score(frames, outputs)
        norm = (255.0 - vfi_score)/255.0
        return norm


    def vfi_score(self, ori_frames, interpolate_frames):
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        scores = []
        for i in range(len(interpolate)):
            scores.append(self.get_diff(ori[i], interpolate[i]))
        return np.mean(np.array(scores))


    def get_diff(self, img1, img2):
        img = cv2.absdiff(img1, img2)
        return np.mean(img)

def extract_dino_features(image_path, processor, model, device='cuda'):
    """提取单张图像的DINO特征(CLS token)"""
    image = Image.open(image_path).convert('RGB')
    with torch.no_grad():
        inputs = processor(images=image, return_tensors='pt').to(device)
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # 使用CLS token
    return features

def extract_video_dino_features(video_path, processor, model, resize=None, device='cuda'):
    """提取视频所有帧的DINO特征"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    frame_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        pil_image = Image.fromarray(frame).convert('RGB')
        with torch.no_grad():
            inputs = processor(images=pil_image, return_tensors='pt').to(device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            frame_features.append(features)
    cap.release()

    return torch.cat(frame_features, dim=0)  # [num_frames, hidden_dim]

def compute_dino_i(video_path, ref_image_folder, model_path, device='cuda'):
    """计算视频与参考图像的DINO相似度"""
    # 加载DINO模型
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    
    # 提取所有参考图的DINO特征
    ref_features = []
    for filename in os.listdir(ref_image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(ref_image_folder, filename)
            features = extract_dino_features(image_path, processor, model, device)
            ref_features.append(features)
    ref_features = torch.cat(ref_features, dim=0)  # [num_ref, hidden_dim]
    
    video_results = []
    for video in video_path:
        # 提取视频帧特征
        video_features = extract_video_dino_features(video, processor, model, device=device) # [num_frames, hidden_dim]
        
        # 计算余弦相似度
        cos_sim = nn.CosineSimilarity(dim=2)
        similarity_matrix = cos_sim(
            video_features.unsqueeze(1),  # [num_frames, 1, hidden_dim]
            ref_features.unsqueeze(0)     # [1, num_ref, hidden_dim]
        )  # -> [num_frames, num_ref]

        # 每帧取与参考集的最大相似度（或平均）
        frame_scores = similarity_matrix.mean(dim=1)    # .max(dim=1)[0]
        final_score = frame_scores.mean().item()
        video_results.append(final_score)
    
    avg_score = np.mean(video_results)
    return avg_score

def extract_clip_features(image_path, processor, model, device='cuda'):
    real_image = processor(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        real_image_embedding = model.encode_image(real_image)
    return real_image_embedding

def extract_video_clip_features(video_path, processor, model, resize=None, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    frame_features = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize:
            frame = cv2.resize(frame, resize)
        pil_image = Image.fromarray(frame).convert("RGB")
        with torch.no_grad():
            pil_image = processor(pil_image)
            frame_embeddings = pil_image.unsqueeze(0).to(device)
            frame_embeddings = model.encode_image(frame_embeddings)
            frame_features.append(frame_embeddings)
    cap.release()

    return torch.cat(frame_features, dim=0)  # [num_frames, hidden_dim]

def compute_clip_i(video_path, ref_image_folder, use_laion400m_e32, device='cuda'):
    # CLIP model
    if use_laion400m_e32:
        print("Using laion400m_e32...")
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', cache_dir='/home/wangwc/project/CogVideoX/finetune/metrics/ip')
    else:
        model, preprocess = clip.load("ViT-B/32")
    model.to(device).eval()

    # 提取所有参考图像的CLIP特征
    ref_features = []
    for filename in os.listdir(ref_image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(ref_image_folder, filename)
            real_image_embedding = extract_clip_features(image_path, preprocess, model, device)
            ref_features.append(real_image_embedding)
    ref_features = torch.cat(ref_features, dim=0)  # [num_ref, hidden_dim]

    video_results = []
    for video in video_path:
        # 提取视频帧特征
        video_features = extract_video_clip_features(video, preprocess, model, device=device) # [num_frames, hidden_dim]

        # 添加归一化操作
        ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        # 计算余弦相似度
        similarity_matrix = (video_features @ ref_features.T) # 直接矩阵乘法

        # 每帧取与参考集的最大相似度（或平均）
        frame_scores = similarity_matrix.mean(dim=1)
        final_score = frame_scores.mean().item()
        video_results.append(final_score)

    avg_score = np.mean(video_results)
    return avg_score

def extract_video_frame_features(video_path, preprocess, resize=None, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, resize)

        pil_image = Image.fromarray(frame).convert("RGB")
        pil_image = preprocess(pil_image)
        frames.append(pil_image)

    cap.release()
    if len(frames) == 0:
        raise ValueError("视频中没有帧可读取。")
    video_tensor = torch.stack(frames)
    video_tensor = video_tensor.to(device)

    return video_tensor

def compute_clip_t(video_path, prompt, device='cuda'):
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()

    video_results = []
    for video, prompt_ in zip(video_path, prompt):
        text_tokens = clip.tokenize(prompt_).cuda()
        frame_tensor = extract_video_frame_features(video_path=video, preprocess=preprocess, device=device)

        with torch.no_grad():
            image_features = model.encode_image(frame_tensor).float() # [F, 512]
            text_features = model.encode_text(text_tokens).float()
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)

        result_clip_t = np.mean(similarity, axis=1).item()
        video_results.append(result_clip_t)

    avg_score = np.mean(video_results)
    return avg_score

def compute_T_Cons(video_path, use_laion400m_e32, device='cuda'):
    if use_laion400m_e32:
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', cache_dir='/home/wangwc/project/CogVideoX/finetune/metrics/ip')
    else:
        model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    cos_sim = nn.CosineSimilarity(dim=1)

    video_results = []
    for video in video_path:
        # 提取视频帧特征
        video_features = extract_video_clip_features(video, preprocess, model, device=device) # [num_frames, hidden_dim]

        # 同时取所有相邻帧对
        frame_pairs1 = video_features[:-1]  # 第0到倒数第二帧
        frame_pairs2 = video_features[1:]   # 第1到最后一帧

        # 先进行特征归一化
        frame_pairs1 = frame_pairs1 / frame_pairs1.norm(dim=-1, keepdim=True)
        frame_pairs2 = frame_pairs2 / frame_pairs2.norm(dim=-1, keepdim=True)

        # 一次性计算所有相邻帧对的相似度
        similarities = cos_sim(frame_pairs1, frame_pairs2)

        result_t_cons = similarities.mean().item()
        video_results.append(result_t_cons)

    avg_score = np.mean(video_results)
    return avg_score

def dynamic_degree(model_path, video_path, device='cuda'):
    args_new = edict({"model":model_path, "small":False, "mixed_precision":False, "alternate_corr":False})
    dynamic = DynamicDegree(args_new, device)

    video_results = []
    for video in video_path:
        score_per_video = dynamic.infer(video)
        video_results.append(score_per_video)
    avg_score = np.mean(video_results)
    return avg_score

def motion_smoothness(config, ckpt, video_path, device='cuda'):
    motion = MotionSmoothness(config, ckpt, device)

    video_results = []
    for video in video_path:
        score_per_video = motion.motion_score(video)
        video_results.append(score_per_video)
    avg_score = np.mean(video_results)
    return avg_score

def get_mp4_videos(directory):
    # 验证目录存在
    if not os.path.isdir(directory):
        raise ValueError(f"目录不存在: {directory}")
    
    # 获取目录下所有文件
    all_files = os.listdir(directory)
    
    # 过滤MP4文件（不区分大小写）
    mp4_files = [
        os.path.abspath(os.path.join(directory, f))  # 转为绝对路径
        for f in all_files
        if f.lower().endswith('.mp4') and           # MP4扩展名检查
           os.path.isfile(os.path.join(directory, f))  # 确保是文件
    ]
    
    # 按文件名排序后返回
    return sorted(mp4_files)

def evaluate(video_path, prompt, image_dataset_folder, dino_model_folder, dd_model_folder, motion_smoothness_module, use_laion400m_e32=True):    
    """DINO-I"""
    result_dino_i = compute_dino_i(video_path=video_path, ref_image_folder=image_dataset_folder, model_path=dino_model_folder)
    print(f"\033[1;31m *******Average DINO-I score*******: {result_dino_i} \033[0m")

    """CLIP-I"""
    result_clip_i = compute_clip_i(video_path=video_path, ref_image_folder=image_dataset_folder, use_laion400m_e32=use_laion400m_e32)
    print(f"\033[1;31m *******Average CLIP-I score*******: {result_clip_i} \033[0m")

    """CLIP-T"""
    result_clip_t = compute_clip_t(video_path=video_path, prompt=prompt)
    print(f"\033[1;31m *******Average CLIP-T score*******: {result_clip_t} \033[0m")

    """T.Cons score """
    result_t_cons = compute_T_Cons(video_path=video_path, use_laion400m_e32=use_laion400m_e32)
    print(f"\033[1;31m *******Average T.Cons score*******: {result_t_cons} \033[0m")

    """DD score """
    result_dd = dynamic_degree(model_path=dd_model_folder, video_path=video_path)
    print(f"\033[1;31m *******Average DD score*******: {result_dd} \033[0m")

    """motion_smoothness score """
    result_ms = motion_smoothness(config=motion_smoothness_module[0], ckpt=motion_smoothness_module[1] ,video_path=video_path)
    print(f"\033[1;31m *******Average motion_smoothness score*******: {result_ms} \033[0m")

    resual_json = {
        "video_path": video_path,
        "prompt": prompt,
        "DINO-I": result_dino_i,
        "CLIP-I": result_clip_i,
        "CLIP-T": result_clip_t,
        "T.Cons": result_t_cons,
        "DD": result_dd,
        "motion_smoothness": result_ms
    }
    return resual_json

def output_func(output_dir, all_results):
    output_json_path = os.path.join(output_dir, "result.json")
    output_radar_path = os.path.join(output_dir, "radar.png")
    csv_path = os.path.join(output_dir, "result.csv")

    # 保存结果为JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    # 保存CSV（排除prompt）
    csv_fields = ['video_id', 'video_dir'] + \
                [k for k in all_results[0].keys() 
                    if k not in ['video_path', 'prompt']]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        
        for res in all_results:
            # 解析视频目录信息
            first_path = os.path.normpath(res['video_path'][0])
            parts = first_path.split(os.sep)
            vid = parts[-2] if len(parts)>=2 else 'root'
            video_dir = os.path.dirname(first_path)
            
            # 构建CSV行数据
            row = {
                'video_id': vid,
                'video_dir': video_dir,
                **{k:v for k,v in res.items() 
                    if k not in ['prompt', 'video_path']}
            }
            writer.writerow(row)

    # 准备雷达图数据
    metrics = [key for key in all_results[0].keys() 
                if key not in ['video_path', 'prompt']]
    video_data = defaultdict(dict)

    # 提取并处理数据
    max_values = {metric: -np.inf for metric in metrics}
    for result in all_results:
        # 获取目录组名称
        first_path = os.path.normpath(result['video_path'][0])
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(first_path)))
        group_name = parent_dir if parent_dir else "root"

        # 记录最大值用于归一化
        for metric in metrics:
            val = result[metric]
            video_data[group_name][metric] = val
            if val > max_values[metric]:
                max_values[metric] = val

    # 归一化处理
    for video in video_data.values():
        for metric in metrics:
            if max_values[metric] != 0:
                video[metric] /= max_values[metric]

    # 字体设置（保持原样）
    font_path = "/home/wangwc/project/CogVideoX/finetune/metrics/font/Times New Roman.ttf"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": font_manager.FontProperties(fname=font_path).get_name(),
        "pdf.fonttype": 42,
        "ps.fonttype": 42
    })

    # 绘图部分保持不变
    plt.figure(figsize=(8, 8), dpi=300)
    ax = plt.subplot(111, polar=True)

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.append(angles, angles[0])

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    line_styles = ['-', '--', '-.', ':'] * 2
    markers = ['o', 's', 'D', '^', 'v', '<']

    for idx, (group_name, group_metrics) in enumerate(video_data.items()):
        values = [group_metrics[metric] for metric in metrics]
        values.append(values[0])
        
        ax.plot(angles, values,
                color=colors[idx%len(colors)],
                linestyle=line_styles[idx%len(line_styles)],
                linewidth=1.5,
                marker=markers[idx%len(markers)],
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=0.8,
                label=group_name)
        
        ax.fill(angles, values, color=colors[idx%len(colors)], alpha=0.07)

    # 坐标轴设置保持不变
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontfamily='serif')

    label_pad = 8
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        rotation = np.degrees(angle)
        if rotation > 90 and rotation <= 270:
            rotation += 180
        label.set_rotation(rotation)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("center")
        label.set_verticalalignment("center")
        x, y = label.get_position()
        label.set_position((x, y - label_pad/72))

    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(color='#e0e0e0', linestyle='--', linewidth=0.7)
    ax.set_yticklabels([])

    for y in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.text(np.pi/2, y, f'{y:.1f}',
                fontsize=9,
                fontfamily='serif',
                color='#666666',
                va='center')

    legend = ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        fontsize=10,
        title='Experimental Groups',
        title_fontsize=11
    )
    legend.get_title().set_fontfamily('serif')

    plt.savefig(output_radar_path, bbox_inches='tight', facecolor='white')
    plt.close()

def load_prompts(file_path: str) -> list[str]:
    """读取存储 prompts 的 txt 文件，返回字符串列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 逐行读取并去除换行符，过滤空行
        return [line.strip() for line in f if line.strip()]
    

"""training image dataset with ref images"""
dog_dataset_path = '/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/dog2_crop/images'
wolf_dataset_path = '/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/wolf_plushie/images'

"""prompt for video"""
prompt_guitar = ["A playful, shiny dog stands on hind legs in a green meadow under a clear blue sky, strumming a small guitar with its paws. It has a happy expression, tongue out, and wagging tail. Grass sways with colorful wildflowers in the breeze. Sunlight casts soft shadows, creating a vibrant, joyful atmosphere as the dog's playing stirs grass and petals."]  
prompt_run = ["A playful dog with shiny fur, energetically running across a lush green meadow under a bright, clear blue sky. The dog has a happy expression with its tongue slightly out, and its tail wagging rapidly. The grass sways gently in the breeze, dotted with colorful wildflowers. The dog's paws kick up small bits of grass and dust as it sprints."]

# TODO.
prompt_test = '/home/wangwc/project/CogVideoX/inference/prompt_dog_5.txt'
# TODO.
video_list_dir = [
"/home/wangwc/project/CogVideoX/finetune/output_adapter/Cogvideox-5b/joint_motion_id/0222/test02_hypernet2-3_output_dim42_gradmask_200/video",
]
# TODO.
output_dir = '/home/wangwc/project/CogVideoX/finetune/output_adapter/Cogvideox-5b/joint_motion_id/0222/test02_hypernet2-3_output_dim42_gradmask_200/metrics'

image_dataset_folder = dog_dataset_path
prompt_list = load_prompts(prompt_test)
print("lenth of prompt list: ", len(prompt_list))
prompt_for_test = [
    prompt[:300].replace(" *", "")
    for prompt in prompt_list
]
videos_list = [
    get_mp4_videos(video_dir_path) for video_dir_path in video_list_dir
]

# model path TODO.
dino_model_folder = '/home/wangwc/project/CogVideoX/finetune/metrics/dinov2-base'

dd_model_folder = '/home/wangwc/project/CogVideoX/finetune/metrics/raft_model/models/raft-things.pth'

motion_smoothness_module = [
    '/home/wangwc/project/CogVideoX/finetune/metrics/amt_model/AMT-S.yaml',
    '/home/wangwc/project/CogVideoX/finetune/metrics/amt_model/amt-s.pth'
]

os.makedirs(output_dir, exist_ok=True)

all_results = []
for vp in videos_list: # 每个VP为video_list，可以计算该dir内所有video的平均值
    resual_json = evaluate(
        video_path=vp,
        prompt=prompt_for_test,
        image_dataset_folder=image_dataset_folder,
        dino_model_folder=dino_model_folder,
        dd_model_folder=dd_model_folder,
        motion_smoothness_module=motion_smoothness_module,
        use_laion400m_e32=True,
        )
    all_results.append(resual_json)

output_func(output_dir, all_results)