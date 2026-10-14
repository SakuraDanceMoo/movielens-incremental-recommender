# MovieLens分布式增量电影推荐系统

按Word路线开展的大数据课程项目。阶段一基于官方MovieLens 32M完成数据检查与全局热门推荐基线；后续阶段保留计划与验收要求。阶段一采用严格时刻限制的训练热门度，以避免按用户时序划分时跨用户未来事件进入特征。

正式数据来源为[GroupLens官方MovieLens 32M ZIP](https://files.grouplens.org/datasets/movielens/ml-32m.zip)。原始数据仅存本地，仓库保存下载脚本、完整性清单、代码、派生结果和真实验证证据。数据使用条件见官方ZIP中的README.txt，不重新发布原始表。

数据派生输出遵循[MovieLens使用条件](docs/DATA_LICENSE.md)。按官方说明引用：F. Maxwell Harper and Joseph A. Konstan, 2015, *The MovieLens Datasets: History and Context*, [DOI:10.1145/2827872](https://doi.org/10.1145/2827872)。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| [阶段一](01_data_baseline/README.md) | 四表检查、统计、80/10/10划分、100负样本、Most Popular、NDCG@10 | completed，本地验收通过；教师脚本对齐待核验 |
| [阶段二](02_similarity_recall/README.md) | 精确相似度与LSH召回 | planned |
| [阶段三](03_distributed_recommendation/README.md) | Spark分布式推荐 | planned |
| [阶段四](04_streaming_incremental/README.md) | 流式回放与增量更新 | planned |
| [阶段五](05_graph_recommendation/README.md) | 图推荐与特征融合 | planned |
| [阶段六](06_integration_ablation/README.md) | 集成与消融，争取超过基线10% | planned |

## 安装与一键复现

使用Python 3.11，建议至少16GB内存，32GB更稳妥；本次运行环境为32GB Windows主机。为原始数据和派生文件预留至少6GB磁盘空间。所有数据路径由脚本位置解析，不依赖本机绝对路径。

Windows PowerShell：

```powershell
python -m venv .venv
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache/pip'
$env:TEMP = Join-Path (Get-Location) '.temp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
New-Item -ItemType Directory -Force $env:TEMP,$env:PIP_CACHE_DIR,$env:MPLCONFIGDIR | Out-Null
& .venv/Scripts/python.exe -m pip install -r requirements.lock.txt
& .venv/Scripts/python.exe -X utf8 run.py
```

Linux/macOS：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
PYTHONUTF8=1 MPLCONFIGDIR=.cache/matplotlib .venv/bin/python run.py
```

下载器读取系统现有HTTPS代理配置，验证TLS证书，使用HTTP Range断点续传并核对响应字节范围。已完成的raw文件不会覆盖；与既有清单不一致立即停止。准备、评估和复现验证都针对全量数据，不是抽样运行。

一键流程：官方数据下载和哈希 → 六项协议测试 → 四表检查、统计、划分和固定候选 → 训练正反馈计数 → 完整基线评价 → 第二次完整评价 → 全量候选重放、Top10指标复算和哈希验证。运行会重建派生产物与日志，raw保持不变。每位无测试正反馈用户保留划分，但不进入指标宏平均。

## 记录与结果

全量基线：**NDCG@10=0.866186，Recall@10=0.845159**，评价191,007位用户。6项协议测试、22项完整验收通过，两次完整评价结果一致。

- [阶段一实验记录](01_data_baseline/logs/experiment_2026-10-14.md)
- [基线指标](01_data_baseline/logs/baseline_metrics.json) · [独立验证](01_data_baseline/logs/verification.json)
- [数据检查](01_data_baseline/logs/data_quality.json) · [描述统计](01_data_baseline/logs/descriptive_statistics.json)
- [Top10结果 gzip CSV](01_data_baseline/data/processed/baseline_top10.csv.gz)
- [原始哈希清单](01_data_baseline/data/raw_manifest.json) · [派生哈希清单](01_data_baseline/data/processed_manifest.json)
- [路线来源与协议选择](docs/roadmap_and_protocol.md) · [AI辅助记录](docs/AI_USE.md)
- 发布验收回执在根目录`logs/`，回执说明验证的内容提交，不把文件快照相同误述为提交历史相同。

大体积train/validation/test Parquet、训练正反馈事件、candidates.npz和split_boundaries.csv保留本地并在Git中忽略，按上述命令可从官方数据重建；哈希清单登记实际生成文件。阶段二到六暂无实现、训练或评价结果。

教师统一脚本尚未提供。本地正确性与可复现性验收不能替代教师程序一致性验收；边界舍入、时间戳并列、采样和热门度时刻定义的差异见协议文档。未来10%提升必须在同一候选与评价协议上比较。
