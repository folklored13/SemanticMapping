import torch
import config


class SemanticMapManager:
    def __init__(self, engine):
        """
        :param engine: 传入初始化好的 CLIPEngine 实例
        """
        self.engine = engine
        self.labels = []  # 存储原始文本标签
        self.index_features = None  # 存储对应的向量索引

    def build_map(self, labels_list):
        """
        构建语义地图索引
        """
        if not labels_list:
            print("[Warning] No labels provided.")
            return

        self.labels = labels_list
        print(f"[Info] Building index for {len(labels_list)} semantic labels...")

        # 调用引擎进行编码
        self.index_features = self.engine.encode_text_list(labels_list, use_prompt=True)
        print("[Info] Index built successfully.")

    def search(self, user_query, top_k=config.DEFAULT_TOP_K):
        """
        根据用户文本检索地图
        """
        if self.index_features is None:
            raise ValueError("Map index is empty. Please call build_map() first.")

        # 去掉一些动词干扰噪声
        clean_query = user_query.lower().replace("find", "").replace("search", "").replace("show me", "").strip()

        # 1. 编码用户查询 (Query通常不需要加 'a photo of'，除非用户只输入一个单词)
        # 这里假设用户输入的是自然语言句子
        query_feature = self.engine.encode_text_list([clean_query], use_prompt=False)

        # 2. 计算相似度
        raw_similarity = self.engine.compute_similarity(query_feature, self.index_features)

        # # 3. 使用 Softmax 归一化分数 (可选，为了更好看) 或者直接用原始分数
        # probs = raw_similarity.softmax(dim=-1).cpu().squeeze(0)

        # 修改：去掉softmax，使用原始相似度
        score = raw_similarity.cpu().squeeze(0)

        # 4. 获取前 K 个结果
        top_probs, top_indices = score.topk(min(top_k, len(self.labels)))

        results = []
        for score, idx in zip(top_probs, top_indices):
            idx = idx.item()
            score = score.item()

            # 因为现在是原始余弦相似度，阈值设为 0.2 ~ 0.25 之间
            # 如果低于阈值，就说明 CLIP 认为根本不像
            if score < 0.22:
                continue

            results.append({
                "label": self.labels[idx],
                "score": score
            })

        return results