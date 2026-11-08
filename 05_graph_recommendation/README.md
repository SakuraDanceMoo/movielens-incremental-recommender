# 阶段五 图推荐模型探索

状态：官方 MovieLens 上的5,000用户图排序实验完成；混合数据评分 RMSE 版本已完成。本页 PPR/ALS 名次融合结果不属于用户提供的数据，多节点图计算也未验证。

使用训练集前5,000位固定候选用户评分不低于4的正反馈，构建用户-电影二部图并运行带重启的 Personalized PageRank。用户类型画像和电影类型作为节点属性归档，PPR Top10 与 ALS Top10 等权 reciprocal-rank 融合。

## 混合数据评分回归（已完成）

脚本 [`scripts/hybrid_rating_regression.py`](scripts/hybrid_rating_regression.py) 使用度数归一化的两跳用户-电影图传播评分，缺失类型依次回退到电影均值和全局均值。时间留出验证集为386,982条，验证 RMSE=**1.063098**。

```powershell
& D:/AgentProject2026/.venv/Scripts/python.exe -X utf8 05_graph_recommendation/scripts/hybrid_rating_regression.py `
  --mode graph `
  --data-dir 'C:/path/to/训练数据集' `
  --output 'C:/path/to/训练数据集/test_pairs_predictions_stage5.csv' `
  --metrics 05_graph_recommendation/logs/hybrid_rating_metrics.json
```

已生成100,000条 `test_pairs` 预测，SHA-256 为 `bbd05a06e61ed539bbd2cf9d39db1211ad919d5f3c780d7d559fbc4b69221b25`。测试表没有评分标签，真实测试 RMSE 不可计算。详情见[混合数据实验记录](logs/experiment_hybrid_2026-11-08.md)和[混合数据指标](logs/hybrid_rating_metrics.json)。预测 CSV 保留在用户数据目录，不上传仓库。

## 历史图排序实验复现入口

```powershell
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 run_phase5.py
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 05_graph_recommendation/scripts/verify_graph.py
```

PPR 使用 restart=0.15、L1 residual tolerance=1e-6。相同5,000用户中，PPR NDCG@10=0.890248，ALS=0.907241，ALS+PPR=0.914802；融合相对热门基线提升5.56%，没有达到项目最终10%目标。PPR 比 ALS 低1.87%，负结果如实保留。详情见[实验记录](logs/experiment_2026-11-08.md)、[运行指标](logs/graph_metrics.json)、[独立验证](logs/verification.json)和[对比图](curves/graph_model_comparison.png)。

图包含17,961个节点、331,435条无向正反馈边。单机 SciPy 实验验证稀疏图算法与固定候选离线排序；不代表分布式图引擎或在线服务延迟。
