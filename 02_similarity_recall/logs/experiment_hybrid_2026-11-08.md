# 阶段二混合评分回归实验记录

日期：2026-11-08（Asia/Hong_Kong）

输入为用户提供的 `hybrid-half-half-latest100k-v1` 混合数据。`ratings.csv` 共31,900,204条评分，`test_pairs.csv` 共100,000行，只有 `userId,movieId,timestamp` 三列。

## 方法

对每位用户按时间戳降序、同时间按 `movieId` 降序留出最新一条评分，且用户至少有6条评分。训练集31,513,222条，验证集386,982条。模型将用户历史评分按电影类型聚合，对目标电影类型取用户类型均值；缺失类型回退到电影均值，再回退到全局均值，预测裁剪到 `[0.5, 5.0]`。

## 结果

验证 RMSE 为 **1.047323**。MovieLens来源留出子集 RMSE=0.991754，合成来源留出子集 RMSE=1.103906。对 `test_pairs.csv` 已生成100,000条预测，文件 SHA-256 为 `c1dbf4fe95ca2b4b3f3ac597ea65bf7c1b4259f00696e7725ac3d62c44358bb0`。

`test_pairs.csv` 没有真实评分列，因此不能计算真实测试 RMSE；该指标只报告 `ratings.csv` 时间留出验证结果。

详细 JSON 指标见 [`hybrid_rating_metrics.json`](hybrid_rating_metrics.json)，预测文件写入用户数据目录，不复制到仓库。
