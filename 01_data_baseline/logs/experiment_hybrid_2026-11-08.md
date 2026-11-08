# 混合评分数据检查与基线

日期：2026-11-08（Asia/Hong_Kong）

## 数据检查

审计用户提供的 `hybrid-half-half-latest100k-v1`。训练评分共31,900,204条，包含MovieLens来源15,950,149条和合成来源15,950,055条；覆盖396,060位用户与142,609部电影。评分范围为0.5至5.0，均值3.540359。`ratings.csv` SHA-256 为 `b1e1b5e22a177ee32a207e058fa22a3f3601cd77f68b4a091625237a43b96815`。

`test_pairs.csv` 有100,000行，字段为 `userId,movieId,timestamp`，MovieLens来源49,953行、合成来源50,047行，无缺失值，也没有真实评分标签。汇总记录显示没有冷启动用户或电影。

详细统计和检查由 [`audit_hybrid_dataset.py`](../scripts/audit_hybrid_dataset.py) 重现，结果见 [`hybrid_data_audit.json`](hybrid_data_audit.json)。训练数据不复制到仓库。

## RMSE基线

与评分模型相同的逐用户时间留出集包含386,982条评分。训练均值基线 RMSE=1.067599；显式评分 ALS rank=16 的 RMSE=0.897475。留出规则、rank比较和子集结果见[阶段六记录](../../06_integration_ablation/logs/rating_prediction_experiment_2026-11-08.md)。

该 RMSE 是训练表内留出验证，不是 `test_pairs` 测试 RMSE，因为测试表没有真实评分。
