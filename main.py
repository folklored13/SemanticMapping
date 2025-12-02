import os
import json
from core_engine import CLIPEngine
from map_manager import SemanticMapManager

QUERY_FILE = "queries.txt"


def run_batch_test(manager, filename):
    if not os.path.exists(filename):
        return
    print(f"\n>>> Reading batch queries from '{filename}'...")
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            q = line.strip()
            if q and not q.startswith("#"):
                print(f"\n[Batch Command]: {q}")
                perform_search(manager, q)
    print("\n>>> Batch test finished.")


def perform_search(manager, query):
    try:
        results = manager.search(query)
        if not results:
            print("  -> No actionable targets found (low similarity).")
        else:
            # 打印符合文档要求的详细信息
            print(f"  -> Found {len(results)} potential target(s):")
            for i, res in enumerate(results):
                obj = res['object']
                score = res['score']

                # 模拟输出：这才是机器人真正需要的（ID 和 坐标）
                print(f"    {i + 1}. [ID: {obj.obj_id}] {obj.label}")
                print(f"       Location: (x={obj.pose['x']}, y={obj.pose['y']}) | Room: {obj.room_type}")
                print(f"       Match Score: {score:.4f} | Data Confidence: {obj.confidence}")

    except Exception as e:
        print(f"  [Error] {e}")


def main():
    # 1. 初始化
    clip_engine = CLIPEngine()
    map_manager = SemanticMapManager(clip_engine)

    # 2. 模拟语义地图数据 (JSON格式，符合文档数据结构)
    # 包含了 ID, Pose, Room, Confidence, Synonyms
    raw_map_data = [
        {
            "id": "obj_001",
            "label": "sink",
            "room_type": "kitchen",
            "pose": {"x": 2.5, "y": 1.0, "theta": 0.0},
            "confidence": 0.98,
            "synonyms": ["washbasin", "water tap"]
        },
        {
            "id": "obj_002",
            "label": "sofa",
            "room_type": "living_room",
            "pose": {"x": 5.0, "y": 3.5, "theta": 1.57},
            "confidence": 0.95,
            "synonyms": ["couch", "settee"]
        },
        {
            "id": "obj_003",
            "label": "toilet",
            "room_type": "bathroom",
            "pose": {"x": 8.2, "y": 2.1, "theta": 3.14},
            "confidence": 0.99,
            "synonyms": ["commode", "wc"]
        },
        {
            "id": "obj_004",
            "label": "plant",
            "room_type": "balcony",
            "pose": {"x": 1.0, "y": 6.0, "theta": 0.0},
            "confidence": 0.60,  # 置信度中等
            "synonyms": ["flower pot"]
        },
        {
            "id": "obj_005",  # 这是一个干扰项，置信度很低
            "label": "ghost object",
            "room_type": "unknown",
            "pose": {"x": 0, "y": 0, "theta": 0},
            "confidence": 0.1,  # 应该被系统自动过滤
            "synonyms": []
        }
    ]

    # 3. 构建索引
    map_manager.build_map(raw_map_data)

    print("\n" + "=" * 50)
    print("      Semantic Layer Search (v0.1)")
    print("      Supports: ID, Pose, Room Context, Synonyms")
    print("=" * 50)

    # 4. 运行批量测试
    run_batch_test(map_manager, QUERY_FILE)

    # 5. 交互模式
    print("\n[Interactive Mode] (Type 'q' to quit)")
    while True:
        user_input = input("Command: ").strip()
        if user_input.lower() in ['q', 'exit']: break
        if not user_input: continue
        perform_search(map_manager, user_input)


if __name__ == "__main__":
    main()