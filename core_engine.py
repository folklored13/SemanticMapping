import torch
import clip
import config  # 导入配置文件


class CLIPEngine:
    def __init__(self):
        print(f"[Info] Loading CLIP model: {config.MODEL_NAME} on {config.DEVICE}...")
        self.device = config.DEVICE
        self.model, self.preprocess = clip.load(config.MODEL_NAME, device=self.device)
        self.model.eval()  # 设置为评估模式

    def encode_text_list(self, text_list, use_prompt=True):
        """
        将文本列表转化为归一化的特征向量
        :param text_list: 字符串列表 ["apple", "banana"]
        :param use_prompt: 是否自动添加 'a photo of' 前缀
        :return: Tensor (N, D)
        """
        if use_prompt:
            # 提示词工程：对于物体标签，加上语境通常效果更好
            processed_texts = [f"a photo of a {t}" for t in text_list]
        else:
            processed_texts = text_list

        # Tokenize
        text_tokens = clip.tokenize(processed_texts, truncate=True).to(self.device)

        # Encode & Normalize
        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features

    def compute_similarity(self, query_features, map_features):
        """
        计算余弦相似度
        :param query_features: (1, D)
        :param map_features: (N, D)
        :return: (1, N) values, indices
        """
        # 矩阵乘法计算相似度 (因为已经归一化了，点积=余弦相似度)
        # similarity = (100.0 * query_features @ map_features.T)
        similarity = query_features @ map_features.T
        return similarity