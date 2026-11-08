# 阶段五混合评分回归实验记录

日期：2026-11-08（Asia/Hong_Kong）

输入为用户提供的 `hybrid-half-half-latest100k-v1` 混合数据。训练评分31,900,204条，按每位用户最新时间评分留出后，训练集31,513,222条，验证集386,982条。

## 方法

以训练评分构建用户-电影二部图，并按电影度数使用 `1/sqrt(degree)` 归一化边权。将用户历史评分沿两跳传播到电影类型，再对目标电影的类型预测评分；缺失类型回退到电影均值，再回退到全局均值，结果裁剪到 `[0.5, 5.0]`。图聚合和预测由 Spark `local[*]` 执行。

## 结果

验证 RMSE 为 **1.063098**。MovieLens来源留出子集 RMSE=1.005184，合成来源留出子集 RMSE=1.121986。对 `test_pairs.csv` 已生成100,000条预测，文件 SHA-256 为 `bbd05a06e61ed539bbd2cf9d39db1211ad919d5f3c780d7d559fbc4b69221b25`。

`test_pairs.csv` 没有真实评分列，因此不能计算真实测试 RMSE；该指标只报告 `ratings.csv` 时间留出验证结果。旧版 PPR/ALS排序实验仍作为官方 MovieLens 历史记录，多节点图运行未验证。

详细 JSON 指标见 [`hybrid_rating_metrics.json`](hybrid_rating_metrics.json)，预测文件写入用户数据目录，不复制到仓库。
