CUDA_VISIBLE_DEVICES= python ./lora_joint.py \
    --model_path "./CogVideoX-5b"\
    --dtype "bfloat16"\
    --generate_type "t2v" \
    --inf_lora_json_path "./inf_lora_bash/lora_inf_0.json" \