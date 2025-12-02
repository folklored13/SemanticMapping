import torch
import config
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ================= 定义符合文档的数据结构 =================
@dataclass
class SemanticObject:
    obj_id: str  # 唯一标识符
    label: str  # 主要标签 (e.g., "cup")
    pose: Dict[str, float]  # 坐标 {x, y, theta}
    room_type: str = "unknown"  # 房间类型
    confidence: float = 1.0  # 检测置信度
    synonyms: List[str] = field(default_factory=list)  # 同义词列表

    # 缓存该对象的搜索文本，避免重复拼接
    _search_text: str = field(init=False)

    def __post_init__(self):
        # 构建用于 CLIP 编码的“丰富描述” (Rich Description)
        # 格式示例: "kitchen sink inside the kitchen. synonyms: washbasin"
        # 这样用户搜 "sink" 或 "washbasin" 或 "kitchen object" 都能匹配
        syn_str = ", ".join(self.synonyms)
        self._search_text = f"{self.label} located in {self.room_type}. synonyms: {syn_str}"


# ========================================================

class SemanticMapManager:
    def __init__(self, engine):
        self.engine = engine
        self.objects: List[SemanticObject] = []  # 存储对象列表
        self.index_features = None  # 存储向量索引

    def build_map(self, raw_data_list: List[Dict]):
        """
        从原始字典数据构建语义地图对象和索引
        """
        self.objects = []
        search_texts = []

        print(f"[Info] Processing {len(raw_data_list)} semantic objects...")

        for item in raw_data_list:
            # 1. 转换数据为对象
            obj = SemanticObject(
                obj_id=item["id"],
                label=item["label"],
                pose=item["pose"],
                room_type=item.get("room_type", "unknown"),
                confidence=item.get("confidence", 1.0),
                synonyms=item.get("synonyms", [])
            )

            # 2. 过滤掉本身置信度太低的数据 (文档提到的"降低幻觉")
            if obj.confidence < config.MAP_DATA_CONFIDENCE_THRESHOLD:
                print(f"  [Skip] Object {obj.obj_id} ignored due to low confidence ({obj.confidence})")
                continue

            self.objects.append(obj)
            # 3. 收集用于编码的描述文本
            # 加上 "a photo of" 前缀有助于 CLIP 理解
            search_texts.append(f"a photo of {obj._search_text}")

        if not self.objects:
            print("[Warning] No valid objects to index.")
            return

        # 4. 批量编码生成索引
        print(f"[Info] Encoding features for {len(self.objects)} objects...")
        self.index_features = self.engine.encode_text_list(search_texts)
        print("[Info] Semantic Map Index built successfully.")

    def search(self, user_query, top_k=config.DEFAULT_TOP_K):
        """
        执行语义检索
        """
        if self.index_features is None:
            raise ValueError("Map index is empty.")

        # 1. 清洗并编码用户查询
        clean_query = user_query.lower().replace("find", "").replace("search", "").strip()

        # 这里的 Prompt 可以简单点，或者也加上 "a photo of"
        query_vec = self.engine.encode_text_list([f"a photo of {clean_query}"])

        # 2. 计算相似度
        scores = self.engine.compute_similarity(query_vec, self.index_features).cpu().squeeze(0)

        # 3. 获取 Top-K
        top_scores, top_indices = scores.topk(min(top_k, len(self.objects)))

        results = []
        for score, idx in zip(top_scores, top_indices):
            idx = idx.item()
            score = score.item()

            # 阈值过滤
            if score < config.CLIP_MATCH_THRESHOLD:
                continue

            obj = self.objects[idx]

            results.append({
                "object": obj,  # 返回整个对象，包含坐标等信息
                "score": score
            })

        return results