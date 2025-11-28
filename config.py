import torch

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 模型名称
# 可选: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50" 等
MODEL_NAME = "ViT-B/32"

# 检索配置
DEFAULT_TOP_K = 3          # 默认返回前几个结果
THRESHOLD_SCORE = 0.20     # 相似度阈值，低于此分数的忽略