# 阶段三 分布式推荐算法实现

状态：旧版官方 MovieLens 隐式反馈 Spark 实验已完成，多节点集群未验证。当前混合数据上的显式评分 Spark ALS 已在[阶段六](../06_integration_ablation/README.md)运行并用于最终预测；本页历史 NDCG 指标不属于混合数据。

混合数据显式 ALS 使用 31,900,204 条评分训练，在每位用户最新评分留出集上比较 rank=16/32，按 RMSE 选择 rank=16（RMSE=0.897475）。实现和复现入口位于[阶段六评分预测脚本](../06_integration_ablation/scripts/rating_prediction.py)，结果位于[评分预测指标](../06_integration_ablation/logs/rating_prediction_metrics.json)。

实现包括 Spark MLlib 隐式反馈 ALS 和基于 Spark SQL 自连接、余弦相似度聚合的 ItemCF。ALS 用 MovieLens 32M 训练划分中评分不低于 4 的交互，使用 25%、50%、100% 嵌套用户规模做扩展性实验；统一在阶段一冻结的测试正样本与100个负样本上计算 NDCG@10、Recall@10。ItemCF 使用冻结候选用户前5000人的训练正反馈工作集。测试标签不进入拟合。

本次执行模式为一台 Windows 主机上的 `local[*]`，不是 HDFS/YARN 多节点集群。因此结果证明 Spark 分布式 API 在单机多核环境下可运行，不证明跨节点扩展性。吞吐量是本机 ALS 拟合吞吐，不是线上推荐服务吞吐。

## 运行

需要 Python 3.11、Java 11、至少 16 GB RAM，建议32 GB。项目 Python 环境放在仓库上一级 `D:\AgentProject2026\.venv`，使用 `uv` 安装锁定依赖。原始数据不上传；应先完成阶段一并生成本地 `train.parquet`、`candidates.npz` 与 `baseline_per_user.csv.gz`。

PowerShell：

```powershell
$env:JAVA_HOME = 'D:\AgentProject2026\.venv\jdk11'
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
$env:PYTHONUTF8 = '1'
..\.venv\Scripts\python.exe -X utf8 run_phase3.py 2>&1 | Tee-Object 03_distributed_recommendation/logs/spark_run.log
```

从仓库父目录新建共享环境并安装依赖：

```powershell
uv venv --python 3.11 .venv
uv pip sync --python .venv/Scripts/python.exe movielens-incremental-recommender/requirements.lock.txt
```

默认训练规模为 `0.25,0.50,1.00`，ALS 参数为 rank=32、迭代8次、regParam=0.1、alpha=1、seed=20260907、10个用户/物品分块；可用 `--scales`、`--rank`、`--iterations`、`--shuffle-partitions` 调整。规模必须以 `1.00` 结束。

## 结果

运行结果记录在 `logs/spark_metrics.json`，环境记录在 `logs/environment.json`，按比例的每用户指标和全量 ALS / 工作集 ItemCF Top10 在 `data/processed/`，规模曲线在 `curves/als_scalability.png`。日志和指标为实测值；若验收值未达到，将在实验记录中如实说明。

- [实验记录](logs/experiment_2026-10-27.md)
- [扩展性曲线](curves/als_scalability.png)
- [运行指标](logs/spark_metrics.json)
- [运行日志](logs/spark_run.log)

ALS 对无训练因子的用户或电影使用零分，保持候选表完整；并列按 movieId 升序，和阶段一排序约定一致。ALS 优于热门基线的声明基于全量固定候选评估。ItemCF 是5000人工作集结果，不能外推全体用户。
