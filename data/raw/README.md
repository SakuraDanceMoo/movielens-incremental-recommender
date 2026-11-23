# 本地混合数据

用户数据迁入 `hybrid/`，原训练数据与已有预测文件逐字节保留，迁移清单见
[`../hybrid_manifest.json`](../hybrid_manifest.json)。默认实际目录为
`D:\AgentProject2026\movielens-incremental-recommender\data\raw\hybrid`。

此目录的 CSV、模型派生缓存和新增预测仅保留本地，不提交 Git。
从用户原始压缩包恢复数据时，解压到 `hybrid/` 并使用清单校验 SHA-256。
新结果写入独立的 `test_pairs_predictions_transformer.csv`，既有 ALS 结果不覆盖。
