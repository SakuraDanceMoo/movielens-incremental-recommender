# 阶段三 分布式推荐算法实现 data/processed

保存 Spark ALS 全量 Top10、各规模每用户指标及5000用户 ItemCF Top10；不保存重复的训练数据或完整模型矩阵。结果随阶段一训练 Parquet 和固定候选集重建。
