# 阶段一 数据检查与基线构建

状态：完整32M本地完成，教师脚本一致性待核验。数据为32,000,204条评分、200,948用户、87,585电影。

四表检查、统计、时间划分、seed=20260907的100负样本和全局热门基线均已执行。191,007位有测试正样本用户上NDCG@10=0.866186，Recall@10=0.845159；9,941位无正样本用户不进入宏平均。热门度只来自严格早于用户推荐时刻的训练正反馈。

从根目录用项目Python执行run.py；安装命令见[根README](../README.md)。独立脚本按download、test_protocol、prepare、clean_metadata、evaluate、evaluate --output-prefix repeat、verify顺序运行。

- [实验记录](logs/experiment_2026-10-14.md) · [数据报告](logs/data_quality_report.md)
- [划分报告](logs/split_report.json) · [候选报告](logs/candidate_report.json)
- [基线指标](logs/baseline_metrics.json) · [22项完整验收](logs/verification.json)
- [Top10结果](data/processed/baseline_top10.csv.gz) · [逐用户指标](data/processed/baseline_per_user.csv.gz)
- [原始清单](data/raw_manifest.json) · [派生清单](data/processed_manifest.json)
- [数据分布](curves/data_distributions.png) · [基线图](curves/baseline_metrics.png)
- [配置](config.json) · [协议](../docs/roadmap_and_protocol.md)

raw数据不上传；大体积Parquet、模型事件、candidates.npz和详细边界CSV留本地并按脚本重建，哈希清单记录实际结果。tags_clean为全时间段元数据，后续模型必须过滤事件时间。数据派生输出遵循[MovieLens使用条件](../docs/DATA_LICENSE.md)。

六项协议测试通过，完整负采样重放与两次评价的科学输出一致。性能数字为单机批处理与本地排序，不代替Spark或线上服务验收。采样NDCG不是全目录NDCG；教师规则不同则重建基线。阶段二至六未开始。
