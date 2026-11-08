# 阶段五实验记录

日期：2026-11-08（Asia/Hong_Kong）

数据：MovieLens 32M，阶段一固定候选集中的前5,000位用户

复现入口：`run_phase5.py`

## 实验设置

使用训练集评分不低于4的正反馈构建无向二部图。图有5,000个用户节点、12,961个电影节点和331,435条边；用户节点附训练历史聚合的19类电影类型计数，电影节点附类型标签。运行 Personalized PageRank，restart probability=0.15、最大100轮、L1 residual tolerance=1e-6。每位用户在同一冻结测试正样本和100负样本候选集上排序。再将PPR Top10与阶段三ALS Top10做等权 reciprocal-rank sum。

运行环境为 Python 3.11.16、NumPy 2.4.6、SciPy 1.17.1，位于 `D:\AgentProject2026\.venv`。在单机稀疏矩阵上执行，不是分布式图计算。

## 结果

PPR平均迭代89.95轮，最大残差8.89e-7，满足1e-6收敛条件。5,000用户 PPR 的 NDCG@10=0.890248、Recall@10=0.858047；同人群阶段三 ALS 为0.907241和0.840255，PPR 的 NDCG低1.87%，但 Recall较高。PPR+ALS 名次融合 NDCG@10=0.914802、Recall@10=0.868567，相对同用户热门基线 NDCG@10=0.866579 提升5.56%。

图模型单独没有超过 ALS；简单融合有所提升，但尚未达到Word要求的最终基线+10%。该负结果和融合结果均保留。PPR 5000个查询约162.46秒，包含单机稀疏矩阵迭代。

## 复算与限制

独立验证脚本复算了两组Top10的名次、唯一性、排序和 NDCG@10 / Recall@10；PPR最大残差满足配置阈值。结果与图结构、节点属性摘要见 [graph_metrics.json](graph_metrics.json)，独立结果见 [verification.json](verification.json)，对比图见 [graph_model_comparison.png](../curves/graph_model_comparison.png)。

范围是确定性5,000用户工作集。没有运行GraphFrames/多节点集群，也未验证其他节点嵌入或社区检测。
