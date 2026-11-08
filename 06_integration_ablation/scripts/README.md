# 阶段六 系统集成与消融实验 scripts

`integrated_experiment.py` 汇总LSH、ALS、PPR和阶段四流式Top10，在共享候选人群上做候选并集、等权 reciprocal-rank 重排及逐模块消融。

从仓库根目录运行：`uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 run_phase6.py`。

独立验证：`uv run --python D:/AgentProject2026/.venv/Scripts/python.exe python -X utf8 06_integration_ablation/scripts/verify_integration.py`。
