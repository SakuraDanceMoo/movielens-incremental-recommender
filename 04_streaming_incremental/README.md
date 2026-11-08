# 阶段四 流式增量处理机制

状态：旧版官方 MovieLens 排序回放实验完成；混合数据评分 RMSE 版本已完成。本页 NDCG、Recall 和延迟结果不属于用户提供的混合数据。

本阶段按事件时间重放阶段一训练集中的正反馈。内存状态包括累计及30天窗口电影热度与用户累计类型画像；每条事件到达时先清除窗口外事件，再更新热度、用户正反馈数和类型计数。每个推荐查询会将watermark推进到查询时点并过期窗口数据。按阶段一测试用户的首个测试时间生成推荐，测试标签不进入状态。混合分数为 `累计正反馈数 + 0.1 × 窗口正反馈数 + 0.2 × 用户对候选类型的平均画像计数占比`，候选仍使用冻结的每用户测试正样本与100个固定负样本。游标在当前内存回放中幂等；持久化检查点与崩溃恢复尚未实现。

## 混合数据评分回归（已完成）

脚本 [`scripts/hybrid_rating_regression.py`](scripts/hybrid_rating_regression.py) 按事件时间回放评分，只使用严格早于查询时间的累计全局、用户和电影均值，权重为 `0.5 / 0.25 / 0.25`。时间留出验证集为386,982条，验证 RMSE=**1.005831**。

```powershell
& D:/AgentProject2026/.venv/Scripts/python.exe -X utf8 04_streaming_incremental/scripts/hybrid_rating_regression.py `
  --data-dir 'C:/path/to/训练数据集' `
  --output 'C:/path/to/训练数据集/test_pairs_predictions_stage4.csv' `
  --metrics 04_streaming_incremental/logs/hybrid_rating_metrics.json
```

已生成100,000条 `test_pairs` 预测，SHA-256 为 `b01535c72d928578e4fd0248518d1a57b73c0a752a3172493c6041819fc06ca8`。测试表没有评分标签，真实测试 RMSE 不可计算。详情见[混合数据实验记录](logs/experiment_hybrid_2026-11-08.md)和[混合数据指标](logs/hybrid_rating_metrics.json)。预测 CSV 保留在用户数据目录，不上传仓库。

## 历史排序回放运行

需要已按阶段一准备数据，并在 `D:\AgentProject2026\.venv` 安装仓库锁文件中的依赖。Windows PowerShell：

```powershell
$env:PYTHONUTF8 = '1'
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 run_phase4.py 2>&1 | Tee-Object 04_streaming_incremental/logs/stream_run.log
uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 04_streaming_incremental/scripts/verify_phase4.py
```

处理代码、窗口长度和画像权重在 `config.json`。实验输出包括完整逐用户指标与 Top10 gzip CSV、延迟曲线、运行指标和状态验证结果。重放会读取本地 `train.parquet`、`candidates.npz` 与官方 `movies.csv`，不会改写阶段一原始或派生文件。

## 结果

实测数据、日期、环境、运行限制和复算结果见 [实验记录](logs/experiment_2026-10-27.md)、[运行指标](logs/stream_metrics.json)、[独立验证](logs/verification.json) 与 [流式延迟分布](curves/streaming_latency.png)。单条事件处理低于100ms的结论只适用于本文记录的单机 Python 内存回放基准，不代表消息队列端到端延迟或生产服务 SLA。当前 NDCG@10 比基线低约0.25%，Recall@10 也略低，质量目标未通过，需继续调整窗口与混合权重并用独立验证集选参。
