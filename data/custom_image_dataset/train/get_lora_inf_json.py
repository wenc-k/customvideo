import os
import json

lora_id_base_path = './lora_id_result'
lora_motion_base_path = './lora_motion_result'
prompt_base_path = "./data/custom_image_dataset/one_part_train"
output_base_path = "./lora_inf_result"

base_path = "/home/wangwc/project/CogVideoX/finetune/data/custom_image_dataset/train/"

case_data = []
for dir_name in os.listdir(base_path):
    part_dir_name = os.path.join(base_path, dir_name)
    if not os.path.isdir(part_dir_name):
        continue
    part_json_path = os.path.join(part_dir_name, "train.json")
    with open(part_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        id_name = os.path.basename(item["instance_data_root_id"])
        motion_name = os.path.basename(item["instance_data_root_motion"])
        output_subdir_name = f"{id_name}_{motion_name}"
        
        prompt_path = os.path.join(prompt_base_path, id_name, "eval_pro_50.txt")
        lora_id_path = os.path.join(lora_id_base_path, id_name)
        lora_motion_path = os.path.join(lora_motion_base_path, motion_name)
        output_path = os.path.join(output_base_path, output_subdir_name)
        case_data.append({
            "prompt_path": prompt_path,
            "lora_id_path": lora_id_path,
            "lora_motion_path": lora_motion_path,
            "output_path": output_path,
        })
print("length of case_data:", len(case_data))
chunk_size = len(case_data) // 5
for i in range(5):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i != 4 else len(case_data)
    result_json_path = os.path.join("/home/wangwc/project/CogVideoX/finetune/inf_lora_bash", f"lora_inf_{i}.json")
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(case_data[start:end], f, ensure_ascii=False, indent=4)