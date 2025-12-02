import torch

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP 模型名称
# 可选: "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50" 等
MODEL_NAME = "ViT-B/32"

# 检索配置
DEFAULT_TOP_K = 3          # 默认返回前几个结果

# CLIP语义相似度阈值0-1
CLIP_MATCH_THRESHOLD = 0.23     # 相似度阈值，低于此分数的忽略

# 地图数据置信度阈值
# 如果地图本身的检测结果不可信 (e.g., confidence < 0.5), 即使匹配了也不返回
MAP_DATA_CONFIDENCE_THRESHOLD = 0.5
