# 阶段六 系统集成与消融实验

状态：本机离线候选融合和三项消融已完成；相对基线提升10%的课程目标未达到，端到端服务P99尚未验证。

本阶段使用阶段二 LSH Top10、阶段三 ALS Top10、阶段五 PPR Top10 与阶段四流式混合 Top10 构造去重候选并集，再用等权 reciprocal-rank sum 重排。评估沿用阶段一前5,000位候选用户的测试正样本和100个固定负样本。各组件都是离线Top10列表，结果衡量共同工作集列表融合。

## 复现

```powershell
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 run_phase6.py
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 06_integration_ablation/scripts/verify_integration.py
```

消融分别移除流式增量列表、图PPR、LSH，并提供 ALS 单模型参照。详细参数见 [config.json](config.json)，指标表见 [ablation_metrics.csv](logs/ablation_metrics.csv)，独立复算见 [verification.json](logs/verification.json)。各配置 Top10 结果见 [ablation_top10.csv.gz](data/processed/ablation_top10.csv.gz)。

完整融合 NDCG@10=0.906646、Recall@10=0.865408，相对同人群热门基线提升4.62%。候选池均值16.64，正样本候选召回92.19%。五种配置及真实消融见[指标表](logs/ablation_metrics.csv)、[实验记录](logs/experiment_2026-11-08.md)、[独立复算](logs/verification.json)和[消融图](curves/integration_ablation.png)。去除流式增量源后 NDCG@10 为0.908725，比完整融合高0.21%；当前增量列表对本融合配置有轻微负贡献。

本阶段验证候选并集和列表重排，没有部署在线服务。重排P99仅覆盖已准备的列表融合和排序，不含 ALS、PPR、LSH、流式源查询、网络和存储。

课程要求全量 NDCG@10 比阶段一基线至少提升10%。本阶段共同工作集提升4.62%，未达到该目标，也不能代替全体用户验收。
