from core_engine import CLIPEngine
from map_manager import SemanticMapManager


def main():
    # 1. 初始化核心引擎 (加载模型)
    # 这一步比较耗时，只做一次
    clip_engine = CLIPEngine()

    # 2. 初始化地图管理器
    map_manager = SemanticMapManager(clip_engine)

    # 3. 模拟语义地图数据 (实际应用中可能来自数据库或文件)
    # 假设机器人知道这些地点的存在
    my_semantic_map = [
        "kitchen sink",  # 厨房水槽
        "coffee machine",  # 咖啡机
        "office desk",  # 办公桌
        "red sofa in living room",  # 客厅红沙发
        "charging station",  # 充电站
        "potted plant",  # 盆栽
        "toilet",  # 厕所
        "meeting room whiteboard",  # 会议室白板
        "laptop on table"
    ]

    # 4. 构建索引
    map_manager.build_map(my_semantic_map)

    print("\n" + "=" * 40)
    print("      Semantic Search System Ready")
    print("      (Type 'q' or 'exit' to quit)")
    print("=" * 40 + "\n")

    # 5. 循环接收用户输入进行测试
    while True:
        user_input = input("User Command: ").strip()

        if user_input.lower() in ['q', 'exit', 'quit']:
            print("Bye!")
            break

        if not user_input:
            continue

        try:
            # 执行搜索
            results = map_manager.search(user_input)

            # 打印结果
            if not results:
                print("  [Result] No matching targets found (low similarity).")
            else:
                print(f"  [Result] Found {len(results)} matches:")
                for i, res in enumerate(results):
                    print(f"    {i + 1}. Target: '{res['label']}' (Confidence: {res['score']:.4f})")

            print("-" * 30)

        except Exception as e:
            print(f"  [Error] {e}")


if __name__ == "__main__":
    main()