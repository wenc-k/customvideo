import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
#import clip
from . import clip
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModel


def get_all_frame_clip(video_path, model, preprocess, resize=None, device='cuda'):
    generated_frame_embeddings = []
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

        frame_embeddings = pil_image.unsqueeze(0).to(device)

        with torch.no_grad():
            frame_embeddings = model.encode_image(frame_embeddings).cpu().numpy()
            generated_frame_embeddings.append(frame_embeddings)
    
    generated_frame_embeddings = np.vstack(generated_frame_embeddings)
    cap.release()
    if len(frames) == 0:
        raise ValueError("视频中没有帧可读取。")

    video_tensor = torch.tensor(np.stack(frames))  # dtype=torch.float32

    video_tensor = video_tensor.to(device)
    return video_tensor, generated_frame_embeddings

def get_all_frame_dino_score(video_path, model, preprocess, image_features1, resize=None, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    # 初始化相似度总和和计数器
    total_similarity = 0
    count = 0
    cos = nn.CosineSimilarity(dim=0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, resize)

        pil_image = Image.fromarray(frame).convert("RGB")
        with torch.no_grad():
            inputs = preprocess(images=pil_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_features2 = outputs.last_hidden_state
            image_features2 = image_features2.mean(dim=1)
        sim = cos(image_features1[0], image_features2[0]).item()
        #sim = (sim + 1) / 2  # 将相似度值归一化到 [0, 1] 范围
        total_similarity += sim
        count += 1
        #print(f'当前real_image和生成视频第{count}帧的相似度值: {sim}')

    if count > 0:
        average_similarity = total_similarity / count
        print(f'当前real_image DINO score: {average_similarity}')
    else:
        print('没有找到生成图像。')
    cap.release()

    return average_similarity

# evaluate CLIP-T and T.cons
def evaluate_for_motion_only(video_path, prompt):
    #CLIP model
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    # ---CLIP-T---
    print("load video from:", video_path)
    frame_tensor, generated_frame_embeddings = get_all_frame_clip(video_path, model, preprocess)
    text_tokens = clip.tokenize(prompt).cuda()
    
    with torch.no_grad():
        image_features = model.encode_image(frame_tensor).float() # [F, 512]
        text_features = model.encode_text(text_tokens).float() 

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)

    result_clip_t = np.mean(similarity, axis=1)
    # --- T.Cons score ---
    temp = []
    for i in range(len(generated_frame_embeddings)-1):
        embedding1 = np.expand_dims(generated_frame_embeddings[i], axis=0)
        embedding2 = np.expand_dims(generated_frame_embeddings[i + 1], axis=0)
        sim = cosine_similarity(embedding1, embedding2)
        temp.append(sim)
    result_t_cons = np.mean(temp)

    return result_clip_t, result_t_cons

def evaluate(video_path, prompt, image_dataset_folder):
    #CLIP model
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    # DINO model
    model_folder = '/home/wangwc/project/CogVideoX/finetune/metrics/dinov2-base'
    processor_dino = AutoImageProcessor.from_pretrained(model_folder)
    model_dino = AutoModel.from_pretrained(model_folder).to("cuda")
    # ---CLIP-T---
    print("load video from:", video_path)
    frame_tensor, generated_frame_embeddings = get_all_frame_clip(video_path, model, preprocess)
    text_tokens = clip.tokenize(prompt).cuda()
    
    with torch.no_grad():
        image_features = model.encode_image(frame_tensor).float() # [F, 512]
        text_features = model.encode_text(text_tokens).float() 

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)

    result_clip_t = np.mean(similarity, axis=1)

    # --- CLIP-I&DINO-I ---
    clip_i = []
    dino_i = []
    temp = []

    for i, filename in enumerate(os.listdir(image_dataset_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(image_dataset_folder, filename)
            real_image = preprocess(Image.open(image_path)).unsqueeze(0).to("cuda")

            with torch.no_grad():
                real_image_embedding = model.encode_image(real_image).cpu().numpy()

            cosine_similarity_scores = cosine_similarity(generated_frame_embeddings, real_image_embedding)
            average_cosine_similarity = np.mean(cosine_similarity_scores)
            print(f"CLIP-I Score of real img_{i+1} and generated video:{average_cosine_similarity}")
            clip_i.append(average_cosine_similarity)

            #DINO-I
            real_image_dino = Image.open(image_path)
            with torch.no_grad():
                inputs1 = processor_dino(images=real_image_dino, return_tensors="pt").to("cuda")
                outputs1 = model_dino(**inputs1)
                image_features1 = outputs1.last_hidden_state
                image_features1 = image_features1.mean(dim=1)
            
            dino_score = get_all_frame_dino_score(video_path, model_dino, processor_dino, image_features1)
            dino_i.append(dino_score)
    
    result_clip_i = np.mean(clip_i)
    result_dino_i = np.mean(dino_i)

    # --- T.Cons score ---
    for i in range(len(generated_frame_embeddings)-1):
        embedding1 = np.expand_dims(generated_frame_embeddings[i], axis=0)
        embedding2 = np.expand_dims(generated_frame_embeddings[i + 1], axis=0)
        
        sim = cosine_similarity(embedding1, embedding2)
        temp.append(sim)
    result_t_cons = np.mean(temp)
    
    return result_clip_t, result_clip_i, result_dino_i, result_t_cons

if __name__ == '__main__':
    # generated video path for test
    video_path = '/home/wangwc/project/CogVideoX/finetune/output_adapter/Cogvideox-5b/joint_motion_id/0206/test01_joint_catvideotxt/test_video_0_A_playful_dog_*_with_shin.mp4'
    # prompt for video
    prompt_guitar = ["A playful, shiny dog stands on hind legs in a green meadow under a clear blue sky, strumming a small guitar with its paws. It has a happy expression, tongue out, and wagging tail. Grass sways with colorful wildflowers in the breeze. Sunlight casts soft shadows, creating a vibrant, joyful atmosphere as the dog's playing stirs grass and petals."]  
    prompt_run = ["Sharp focus on a playful dog with glossy fur sprinting across a sunlit meadow. Clear unobstructed view captures its wagging tail, lolling tongue, and paws kicking up grass blades. Vibrant wildflowers dot the lush green field under a bright blue sky, sunlight freezing motion details with crisp shadows."]
    prompt_wolf_run = ["a wolf plushie running in the forest"]

    prompt = "A playful dog with shiny fur, sitting upright on its hind legs in a lush green meadow under a bright, clear blue sky, strumming a small guitar with its front paws. The dog has a happy expression, its tongue slightly out, and its tail wagging enthusiastically. The grass sways gently in the breeze"
    # training image dataset with real subject
    dog_dataset_path = '/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/dog2_crop/images'
    wolf_dataset_path = '/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/wolf_plushie/images'

    image_dataset_folder = dog_dataset_path

    result_clip_t, result_clip_i, result_dino_i, result_t_cons = evaluate(
        video_path=video_path,
        prompt=prompt,
        image_dataset_folder=image_dataset_folder,
        )
    
    print('*******Average CLIP-T score*******:', result_clip_t)
    print('*******Average CLIP-I score*******:', result_clip_i)
    print('*******Average DINO-I score*******:', result_dino_i)
    print('*******Average T.Cons score*******:', result_t_cons)