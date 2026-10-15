# 阶段二 相似度计算与召回策略 curves

- `phase2_metrics.png`：Most Popular、ItemCF、UserCF、LSH 的 NDCG@10、查询延迟和邻居重合度。
- `lsh_precision_efficiency.png`：四组 LSH 参数的查询加速比—邻居 Recall@50 曲线；点大小表示平均同桶候选池，虚线为 Word 的 10×与 90%目标。

两张图均由 `scripts/phase2_experiment.py --sweep` 根据实际运行指标生成。
