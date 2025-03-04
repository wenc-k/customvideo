
CUDA_VISIBLE_DEVICES=3 python score_calculate.py \
    --edit_path /private/task/linyijing/CogVideoX1_5/finetune/output-lora/object365/20250126-512-last-add-same-noise-last-loss-lr-2e-5-bs48-26w-length2048-lora_r256_alpha256/checkpoint-4000/test/editing \
    --gt_path_1  /private/task/linyijing/CogVideoX1_5/Metric/full_test_gt/full_gt_1 \
    --gt_path_2 /private/task/linyijing/CogVideoX1_5/Metric/full_test_gt/full_gt_2 \
