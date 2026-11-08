# 阶段五 图推荐模型探索

状态：本机5,000用户图实验完成。PPR 单独低于同工作集 ALS；ALS+PPR 名次融合有提升但未达到10%。多节点图计算未验证。

使用训练集前5,000位固定候选用户评分不低于4的正反馈，构建用户-电影二部图并运行带重启的 Personalized PageRank。用户类型画像和电影类型作为节点属性归档，PPR Top10 与 ALS Top10 等权 reciprocal-rank 融合。

复现入口：

```powershell
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 run_phase5.py
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 05_graph_recommendation/scripts/verify_graph.py
```

PPR 使用 restart=0.15、L1 residual tolerance=1e-6。相同5,000用户中，PPR NDCG@10=0.890248，ALS=0.907241，ALS+PPR=0.914802；融合相对热门基线提升5.56%，没有达到项目最终10%目标。PPR 比 ALS 低1.87%，负结果如实保留。详情见[实验记录](logs/experiment_2026-11-08.md)、[运行指标](logs/graph_metrics.json)、[独立验证](logs/verification.json)和[对比图](curves/graph_model_comparison.png)。

图包含17,961个节点、331,435条无向正反馈边。单机 SciPy 实验验证稀疏图算法与固定候选离线排序；不代表分布式图引擎或在线服务延迟。
