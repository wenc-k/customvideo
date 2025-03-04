import os, sys

import open_clip
sys.path.append(os.getcwd())
import glob
from PIL import Image
import albumentations
import numpy as np
import torchvision
import torch
from tqdm import tqdm, trange
import clip
from torchvision import transforms
from test_prompt import prompts_test, prompts_dict, prompts_tes_name, instance_dict, instance_dict_cn

class CLIPEvaluator(torch.nn.Module):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__()
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        print(clip_preprocess)
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(images.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix

class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

class DinoImageEncoder(torch.nn.Module):
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]
    
    def __init__(self, model_name, encode_type="last", image_size=480, version=1):
        super().__init__()
        self.model_name = model_name
        if version == 1:
            self.model = torch.hub.load('/private/task/linyijing/.cache/torch/hub/facebookresearch_dino_main', model=model_name, pretrained=True, source='local') #TODO.
        else:
            self.model = torch.hub.load('/private/task/linyijing/.cache/torch/hub/facebookresearch_dinov2_main', model=model_name, pretrained=True, source='local') #TODO.
            #self.model = torch.hub.load("/home/zdmaogroup/chennan/.cache/torch/hub/facebookresearch_dinov2_main/",model=model_name, pretrained=True, source='local')
        #self.model.load_state_dict(torch.load('/home/cn/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth'))
        self.model.eval()  
        self.encode_type = encode_type
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in DinoImageEncoder.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()
        
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = transforms.Resize((image_size, image_size))
        self.global_transform = transforms.Compose([global_resize_transform, imagenet_norm])

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output
    
    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[DinoImageEncoder.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature
    
    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature
    
    # input raw images
    def encode(self, input_img):
        input_img = self.global_transform(input_img)
        if self.encode_type == "full":
            return self.get_feature_from_input(input_img)
        elif self.encode_type == "last":
            return self.get_feature_from_input(input_img)[-1]
        elif self.encode_type == "qkv":
            return self.get_qkv_feature_from_input(input_img)
        elif self.encode_type == "attn":
            return self.get_attn_feature_from_input(input_img)
        else:
            raise NotImplementedError()  
  
trans = albumentations.Compose([
                albumentations.Resize(height=512, width=512),
            ])

def crop_image(image_path=None,img=None):
    if img==None:
        img=Image.open(image_path)
    assert img.size==(2048,512)
    target_width, target_height = 512, 512
    img_list=[]
    # 切割图像并保存
    for i in range(4):
        left = i * target_width
        right = left + target_width
        top = 0
        bottom = top + target_height
        # 从原图中裁切出新图像
        cropped_image = img.crop((left, top, right, bottom))
        img_list.append(cropped_image)
        #对每个切割后的图像与原始图像进行对比
        # save_path = image_path.replace('.png', '')
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # save_path = save_path + '/' + str(i) + '.png'
        # cropped_image.save(save_path)
    return img_list

class CLIPDirEvaluator_image(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-L/14') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images):
        sim_samples_to_img  = self.img_to_img_similarity(src_images, gen_samples)
        return sim_samples_to_img

class CLIPDirEvaluator_text(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-L/14') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, text, generated_images):
        text_alignment  = self.txt_to_img_similarity(text, generated_images)
        return text_alignment
    
class DINODirEvaluator(torch.nn.Module):
    def __init__(self, model_name="dino_vits16", model_version=1) -> None:
        super().__init__()
        self.dino_extracter = DinoImageEncoder(model_name=model_name, encode_type="last", image_size=512, version=model_version)
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        
    def evaluate(self, gen_samples, src_images):
        gen_feats = self.dino_extracter.encode(gen_samples)
        src_feats = self.dino_extracter.encode(src_images)
        
        similarity = self._cosine_similarity(gen_feats, src_feats)
        similarity_mean = similarity.mean(-1)
        
        return similarity_mean    

class OpenCLIPDirEvaluator_image(torch.nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device
        model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32', cache_dir='/home/wangwc/project/CogVideoX/finetune/metrics/ip')
        self.model = model.to(self.device)
        self.clip_preprocess = clip_preprocess
        print(clip_preprocess)
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor
        
    def evaluate(self, gen_samples, src_images):
        gen_samples = self.preprocess(gen_samples).to(self.device)
        src_images = self.preprocess(src_images).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            gen_feats = self.model.encode_image(gen_samples)
            src_feats = self.model.encode_image(src_images)
            gen_feats /= gen_feats.norm(dim=-1, keepdim=True)
            src_feats /= src_feats.norm(dim=-1, keepdim=True)
            similarity = (src_feats @ gen_feats.T).mean()
        return similarity
    
    
def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = trans(image = image)
    image = Image.fromarray(image["image"])
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    return image

def postprocessing(image):
    image = np.array(image)
    image = trans(image = image)
    image = Image.fromarray(image["image"])
    image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
    return image


def clip_alignment_image_function(
        generation_path, 
        groundtruth_path="", 
        sample_num=1,
        clip_model="ViT-L/14",
    ):
    print("begin clip image alignment")
    parent_dir = os.path.dirname(os.path.dirname(generation_path))
    metric_save_path = os.path.join(parent_dir, "metrics.txt")
    evaluator = CLIPDirEvaluator_image(device="cuda", clip_model=clip_model)
    similarity = []    
    images_paths = glob.glob("{}/*.jpg".format(groundtruth_path))

    prompts_list = os.listdir(generation_path)
    for prompt in prompts_list:
        generation_sub_path = os.path.join(generation_path, prompt) 
        for i in trange(len(images_paths)): 
            if sample_num == 1:
                real_image = read_image(images_paths[i])
                name = images_paths[i].split("/")[-1].split('.')[0]
                sub_name =  name+".png"
                gen_path = generation_sub_path + "/" + sub_name
                # if "03.png" in gen_path:
                #     continue
                fake_image = Image.open(gen_path).convert("RGB")
                fake_image_list = crop_image(img=fake_image)
                sim=0
                for fake_image in fake_image_list:
                    fake_image = postprocessing(fake_image)
                    sim+= evaluator.evaluate(gen_samples=fake_image.cuda(), src_images=real_image.cuda()).cpu().item()
                    #print(sim)
                sim=sim/4
            else:
                real_image = read_image(images_paths[i]).expand(sample_num, -1, -1, -1)
                path_list = images_paths[i].split("/")
                sub_name = path_list[-2] + "/" + path_list[-1].replace(".jpg", "")
                gen_paths = generation_sub_path + sub_name + "/"
                gen_paths_list = glob.glob("{}/*.png".format(gen_paths))
            
                fake_image = []
                for j in range(sample_num):
                    fake_image.append(read_image(gen_paths_list[j]))
                fake_image = torch.cat(fake_image, dim=0)
                real_image = real_image.expand(sample_num, -1, -1, -1)
                sim = evaluator.evaluate(gen_samples=fake_image.cuda(), src_images=real_image.cuda())        
            similarity.append(sim)
        
    print("clip image alignment of {}: {}".format(clip_model, np.array(similarity).mean()))
    with open(metric_save_path, "a+") as f:
        f.write('\n')
        f.write('generation path: {} \n'.format(generation_path))
        f.write('clip image alignment of {} : {} \n'.format(clip_model, np.array(similarity).mean()))

def dino_alignment_function(
        generation_path, 
        groundtruth_path, 
        sample_num=1,
        model_name="dino_vits8",
        model_version=1,
    ):
    print("begin dino image alignment")
    parent_dir = os.path.dirname(os.path.dirname(generation_path))
    metric_save_path = os.path.join(parent_dir, "metrics.txt")
    evaluator = DINODirEvaluator(model_name=model_name, model_version=model_version).cuda()
    
    similarity = []
    images_paths = glob.glob("{}/*.jpg".format(groundtruth_path))
    prompts_list = os.listdir(generation_path)
    for prompt in prompts_list:
        generation_sub_path = os.path.join(generation_path, prompt)
        for i in trange(len(images_paths)): 
            if sample_num == 1:
                real_image = read_image(images_paths[i])
                name = images_paths[i].split("/")[-1].split('.')[0]
                sub_name = name +".png"
                gen_path = generation_sub_path + "/" + sub_name
                # if "03.png" in gen_path:
                #     continue
                fake_image = Image.open(gen_path).convert("RGB")
                fake_image_list=crop_image(img=fake_image)
                sim = 0
                for fake_image in fake_image_list:
                    fake_image=postprocessing(fake_image)
                    sim+= evaluator.evaluate(gen_samples=fake_image.cuda(), src_images=real_image.cuda()).cpu().item()
                    #print(sim)
                sim=sim/4
            else:
                real_image = read_image(images_paths[i]).expand(4, -1, -1, -1)
                path_list = images_paths[i].split("/")
                sub_name = path_list[-2] + "/" + path_list[-1].replace(".jpg", "")
                gen_paths = generation_sub_path + sub_name + "/"
                gen_paths_list = glob.glob("{}/*.png".format(gen_paths))
            
                fake_image = []
                for j in range(sample_num):
                    fake_image.append(read_image(gen_paths_list[j]))
                fake_image = torch.cat(fake_image, dim=0)
                real_image = real_image.expand(sample_num, -1, -1, -1)
                sim = evaluator.evaluate(gen_samples=fake_image.cuda(), src_images=real_image.cuda())
            similarity.append(sim)
        
    print("dino image alignment: {}".format(np.array(similarity).mean()))
    with open(metric_save_path, "a+") as f:
        f.write('generation path: {} \n'.format(generation_path))
        f.write('dino image alignment, model_name {}, version {}: {} \n'.format(model_name, model_version, np.array(similarity).mean()))
        f.write('\n')

def clip_alignment_text_function(generation_path, batch_size=1, clip_model="ViT-L/14"):
    print("begining clip text alignment evaluation")
    parent_dir = os.path.dirname(os.path.dirname(generation_path))
    metric_save_path = os.path.join(parent_dir, "metrics.txt")
    evaluator = CLIPDirEvaluator_text(device="cuda", clip_model=clip_model).cuda()

    prompts_list = os.listdir(generation_path)
    print(prompts_list)
    print("text prompt number: ", len(prompts_list))
    alignment_list = []
    for prompt in prompts_list:
        complete_prompt = prompts_dict[prompt]
        category_list = os.listdir(generation_path + "/" + prompt)
        print(prompt)
        for category in tqdm(category_list, total=len(category_list)):
            prompt_with_category = complete_prompt.replace("{}", instance_dict[category.split('.')[0]])
            gen_path =  generation_path + "/" + prompt + "/" + category
            # if "03.png" in gen_path:
            #         continue
            fake_image = Image.open(gen_path).convert("RGB")
            fake_image_list = crop_image(img=fake_image)
            alignment=0
            for fake_image in fake_image_list:
                fake_image=postprocessing(fake_image)
                alignment += evaluator.evaluate(text=prompt_with_category, generated_images=fake_image.cuda()).cpu().item()
            alignment=alignment/4
            print(alignment)
            alignment_list.append(alignment)
    print("clip text alignment of {}: {}".format(clip_model, np.array(alignment_list).mean()))
    with open(metric_save_path, "a+") as f:
        f.write('generation path: {} \n'.format(generation_path))
        f.write('clip text alignment of {}: {} \n'.format(clip_model, np.array(alignment_list).mean()))
   
def open_clip_image_alignment_function(generation_path, groundtruth_path):
    print("begin clip image alignment")
    parent_dir = os.path.dirname(os.path.dirname(generation_path))
    metric_save_path = os.path.join(parent_dir, "metrics.txt")
    evaluator = OpenCLIPDirEvaluator_image(device="cuda")
    similarity = []    
    images_paths = glob.glob("{}/*.jpg".format(groundtruth_path))

    prompts_list = os.listdir(generation_path)
    for prompt in prompts_list:
        generation_sub_path = os.path.join(generation_path, prompt) 
        for i in trange(len(images_paths)): 
            real_image = read_image(images_paths[i])
            name = images_paths[i].split("/")[-1].split('.')[0]
            sub_name =  name+".png"
            gen_path = generation_sub_path + "/" + sub_name
            # if "03.png" in gen_path:
            #     continue
            # fake_image = Image.open(gen_path).convert("RGB")
            # fake_image_list = crop_image(img=fake_image)
            fake_image_list = crop_image(image_path=gen_path)
            sim=0
            for fake_image in fake_image_list:
                fake_image = postprocessing(fake_image)
                sim+= evaluator.evaluate(gen_samples=fake_image.cuda(), src_images=real_image.cuda()).cpu().item()
                #print(sim)
            sim=sim/4
            similarity.append(sim)
            # real_image = read_image(images_paths[i]).expand(4, -1, -1, -1)
            # path_list = images_paths[i].split("/")
            # sub_name = path_list[-2].replace('full_gt_2', '') + "/" + path_list[-1].replace(".jpg", "")
            # gen_paths = generation_sub_path + sub_name + "/"
            # gen_paths_list = glob.glob("{}/*.png".format(gen_paths))
            # print(gen_paths)
            # fake_image = []
            # for j in range(4):
            #     sim = evaluator.evaluate(gen_samples=gen_paths_list[j], src_images=images_paths[i])        
            #     similarity.append(sim)
    print("open-clip image alignment: {}".format(np.array(similarity).mean()))
    with open(metric_save_path, "a+") as f:
        f.write('\n')
        f.write('generation path: {} \n'.format(generation_path))
        f.write('open-clip image alignment: {} \n'.format(np.array(similarity).mean()))
        
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--edit_path", type=str, default="/private/task/linyijing/CogVideoX1_5/finetune/output-lora/object365/20250129-512-last-add-same-noise-last-loss-lr-1e-4-bs40-26w-length2048-lora_r256_alpha256-sepsrate_adaln/checkpoint-4000/test/editing")
    parser.add_argument("--gt_path_1", type=str, default="/private/task/linyijing/CogVideoX1_5/Metric/full_test_gt/full_gt_1")
    parser.add_argument("--gt_path_2", type=str, default="/private/task/linyijing/CogVideoX1_5/Metric/full_test_gt/full_gt_2")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument("--dino_model", type=str, default="dino_vits8")
    parser.add_argument("--dino_model_version", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":    
    args = parse_args()
    # clip_alignment_image_function(generation_path=args.edit_path, groundtruth_path=args.gt_path_2, clip_model=args.clip_model)
    # dino_alignment_function(generation_path=args.edit_path, groundtruth_path=args.gt_path_2, model_name=args.dino_model, model_version=args.dino_model_version)
    # clip_alignment_text_function(generation_path = args.edit_path, clip_model=args.clip_model)
    open_clip_image_alignment_function(generation_path=args.edit_path, groundtruth_path=args.gt_path_2)