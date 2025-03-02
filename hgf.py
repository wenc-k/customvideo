from datasets import load_dataset

# 方法 1：通过环境变量传递 Token（推荐）
import os
os.environ["HF_TOKEN"] = "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"  # 替换为你的 Token

dataset = load_dataset("inserk/lora_id_result", repo_type="dataset")

# 方法 2：直接在函数中传递 Token
dataset = load_dataset(
    "inserk/lora_id_result",
    repo_type="dataset",
    token="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"  # 替换为你的 Token
)