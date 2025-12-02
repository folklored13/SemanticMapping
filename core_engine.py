import torch
import clip
import config


class CLIPEngine:
    def __init__(self):
        print(f"[Info] Loading CLIP model: {config.MODEL_NAME} on {config.DEVICE}...")
        self.device = config.DEVICE
        self.model, self.preprocess = clip.load(config.MODEL_NAME, device=self.device)
        self.model.eval()

    def encode_text_list(self, text_list):
        """
        将文本描述转化为特征向量
        注意：这里不再自动加 "a photo of"，因为我们会在 map_manager 里构建更复杂的描述
        """
        # Tokenize (截断长度，防止描述过长报错)
        text_tokens = clip.tokenize(text_list, truncate=True).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(text_tokens)
            features /= features.norm(dim=-1, keepdim=True)

        return features

    def compute_similarity(self, query_features, map_features):
        """
        计算余弦相似度 (0~1)
        """
        return query_features @ map_features.T