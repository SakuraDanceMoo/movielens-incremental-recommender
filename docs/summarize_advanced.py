"""Update project navigation and measured experiment records after audit is opened."""
from __future__ import annotations
import json,platform,sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT/'06_integration_ablation';LOGS=STAGE/'logs/advanced'


def main():
    evaluation=json.loads((LOGS/'audit_evaluation.json').read_text());rows={r['name']:r for r in evaluation['rows']}
    selected=evaluation['selected'];result=json.loads((LOGS/f'result_{selected}.json').read_text());new=rows[selected];old=rows['previous_transformer_ensemble'];als=rows['original_als']
    rel=(old['audit_rmse']-new['audit_rmse'])/old['audit_rmse']*100
    relals=(als['audit_rmse']-new['audit_rmse'])/als['audit_rmse']*100
    names=['mf64_r003','mf64_r03','mf64_r08','mf128_r04','svdpp32_r03','memory32_time_r03','memory32_notime_r03','mf64_time_r03']
    trials=[]
    for n in names:
        r=json.loads((LOGS/f'result_{n}_validation.json').read_text());trials.append({'name':n,'best_epoch':r['best_epoch'],'development_rmse':r['development_rmse'],'fit_seconds':r['seconds'],'parameters':r['parameters'],**{k:r['config'].get(k,False) for k in ['mode','dim','reg','temporal']}})
    pl.DataFrame(trials).write_csv(LOGS/'neural_trials.csv')
    environment={'python_executable':sys.executable,'python':platform.python_version(),'platform':platform.platform(),
                 'packages':{n:version(n) for n in ['torch','numpy','polars','catboost','pandas','matplotlib']},
                 'gpu':'NVIDIA GeForce RTX 4070 Ti 12GB','dependency_root':'D:/AgentProject2026/.venv',
                 'neural_seed':20261123,'calibration_seeds':[20261123,20261124],'catboost_threads':8}
    (LOGS/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    config={'stage':6,'data_dir':'data/raw/hybrid','history_length':16,'history_rule':'strict timestamp prefix; exclude ties',
            'fitting_rows':31513222,'outer_holdout_rows':386982,'development_rows':309513,'audit_rows':77469,
            'selected':selected,'features':result['features'],'calibration_params':result['params'],
            'epoch_rule':'development-only minimum, refit same epoch count on full ratings','calibration_rule':'five-fold OOF in development; five-model average in audit/test',
            'source_primary_metadata':'../docs/advanced_sources.json','output':'data/raw/hybrid/test_pairs_predictions_advanced.csv','test_labels_available':False}
    (STAGE/'advanced_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    ci=evaluation['audit_paired_old_minus_new_rmse_ci95']
    record=f'''# 阶段六 历史记忆与残差校正实验

日期为2026-11-23，时区Asia/Hong_Kong。使用D盘共享环境，原始混合数据31,900,204条，12个既有文件保持原字节。近期候选条件记忆思想做小型评分回归改编，完整方法和论文边界见[方法说明](../../../docs/advanced_method.md)。

## 核心结果与评价口径

固定本轮开发选择后，同一77,469位审计用户上，原ALS为 **{als['audit_rmse']:.9f}**，上一轮Qwen/MF/ALS融合为 **{old['audit_rmse']:.9f}**，本轮为 **{new['audit_rmse']:.9f}**。相对上一轮降低 **{rel:.3f}%**，相对原ALS降低 **{relals:.3f}%**。旧减新RMSE的配对bootstrap 95%区间为[{ci[0]:.6f}, {ci[1]:.6f}]，1000次用户重采样，条件于已固定模型。

本轮开发五折OOF为 **{new['development_rmse']:.9f}**。386,982条外层留出合并结果为 **{new['overall_rmse']:.9f}**，其中开发行用OOF，审计行用五模型均值。上一轮0.875872445是原外层纯验证选模成绩，两轮第二层监督不同，严格比较优先使用审计的同一批用户。旧项目模型曾用全体外层留出选模，因此审计仅对本轮新校正保留，仍需全新隐藏测试。

## 实际训练和配置

本轮训练8个神经基础配置，再按开发最佳轮次全量重训8个。候选注意力读取最近16条严格过去记录；强正则MF、时间偏置MF和截断SVD++均保留。批量16,384，AdamW，种子20261123。完整入模75特征，排除任意坐标方向的32个隐向量维度。

最终选中 `{selected}`，{result['feature_count']}列，CatBoost深度{result['params']['depth']}、{result['params']['iterations']}轮、学习率0.035、叶正则25、子采样0.8、8线程。五折只在309,513位开发用户内训练，每条开发预测未使用本条标签拟合校正器。没有审计eval_set或早停。

## 匹配消融与负结果

| 配置 | 开发OOF RMSE | 审计RMSE |
| --- | ---: | ---: |
'''
    labels={'calibration_v2_d8':'完整75特征','calibration_v2_d8_no_history':'移除历史特征','calibration_v2_d8_no_neural':'移除本轮神经特征','calibration_v2_d8_no_semantics':'移除额外语义特征'}
    for n,label in labels.items():record+=f"| {label} | {rows[n]['development_rmse']:.9f} | {rows[n]['audit_rmse']:.9f} |\n"
    record+='''
历史和本轮神经特征有用；额外语义特征移除后开发成绩更好，最终采用该简化方案。语义移除仍保留旧Qwen分支，不是对整个系统删除Qwen。树模型的特征组消融不能当作单个注意力算子的独立因果证明。

加大MF维度没有超过较小的时间MF。训练损失下降而开发误差回升的实际曲线保留，不延长训练冒充提升。早期66维pilot使用过中途的时间MF预测快照，与最终特征不完全相同，仅保留为探索记录，不进入正式比较。

额外树种子仅检验校正层随机性，全部新基础网络仍是一个种子。原来Qwen轮的两种子结果不等同于本轮完整流程重复实验。

## 全量预测与复核

基础模型按固定轮次在全部31,900,204条评分上重训，校正器使用冻结的五折模型。新CSV保持原测试配对顺序和字段，新增predicted_rating，共100,000行。旧预测保留。原始表和模型不提交GitHub，代码、配置、日志、真实曲线与验收记录发布。

14项协议行为测试见[protocol_tests.json](protocol_tests.json)，全数据及预测核验见[verification.json](verification.json)，新CSV字节哈希见[prediction_manifest.json](prediction_manifest.json)。测试没有真实评分，测试RMSE不可计算。全局离线统计使用整个允许拟合表，只有历史记忆满足逐查询严格时间前缀。

图表分别引用真实实验JSON，结构图和协议图明确是示意。系统仍为离线评分流程，没有多节点服务或线上延迟证据。
'''
    (LOGS/'experiment_2026-11-23.md').write_text(record,encoding='utf-8')
    summary=f'''## 2026-11-23 继续优化：候选历史记忆与五折残差校正

本轮预留的77,469位审计用户上，上一轮融合 **{old['audit_rmse']:.6f} → {new['audit_rmse']:.6f}**，相对降低 **{rel:.3f}%**；原ALS在相同审计用户上为{als['audit_rmse']:.6f}。开发OOF为{new['development_rmse']:.6f}，外层留出合并OOF/审计为{new['overall_rmse']:.6f}。后者含第二层开发监督，不能直接当作与旧0.875872完全相同训练协议的独立测试比较。

采用近期候选条件记忆思想的评分回归改编，结合时间MF、截断SVD++、历史统计和CatBoost残差校正。根据开发集选择{result['feature_count']}特征方案，移除额外语义特征反而更好。没有完整复现工业TransX/LWR，也不宣称全球SOTA。固定方案后才打开本轮审计；旧项目曾用全体外层留出调参，所以仍需全新隐藏测试。

![同一审计用户比较](docs/reports/assets/advanced/audit_comparison.png)

新预测在本地 `data/raw/hybrid/test_pairs_predictions_advanced.csv`。训练数据、依赖与模型仍在D盘；没有阶段七。[方法与复现边界](docs/advanced_method.md)、[正式实测与消融](06_integration_ablation/logs/advanced/experiment_2026-11-23.md)、[全部同人群指标](06_integration_ablation/logs/advanced/advanced_comparison.csv)、[独立核验](06_integration_ablation/logs/advanced/verification.json)。测试无标签，没有测试RMSE。

'''
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    if '## 2026-11-23 继续优化' in text:text=text[:text.index('## 2026-11-23 继续优化')]+text[text.index('## 2026-11-23 更新：'):]
    text=text.replace('## 2026-11-23 更新：Qwen3 语义与 Transformer 优化','## 2026-11-23 更新：Qwen3 语义与 Transformer 优化（上一轮归档）')
    idx=text.index('## 2026-11-23 更新：');text=text[:idx]+summary+text[idx:]
    text=text.replace('最新预测为本地','上一轮预测为本地')
    lines=text.splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| [阶段六]'):lines[i]=f'| [阶段六](06_integration_ablation/README.md) | 集成、消融与最终评分预测 | 本轮审计RMSE={new["audit_rmse"]:.6f}，外层合并OOF/审计={new["overall_rmse"]:.6f}；匹配消融与100,000条预测完成 |'
    readme.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    stage=STAGE/'README.md';body=stage.read_text(encoding='utf-8')
    body=body.replace('## 上一轮归档：冻结 Qwen3','## 当前方案：冻结 Qwen3')
    anchor='## 当前方案：冻结 Qwen3'
    if '## 本轮方案' in body:body=body[:body.index('## 本轮方案')]+body[body.index(anchor):]
    idx=body.index(anchor)
    block=summary.replace('## 2026-11-23 继续优化：候选历史记忆与五折残差校正','## 本轮方案：候选历史记忆与五折残差校正').replace('](docs/','](../docs/').replace('](06_integration_ablation/','](')
    block+='''### 本轮复现入口

先完成上一轮数据与基础分支准备。所有命令在仓库根目录执行，先设置D盘缓存环境变量（见下文依赖段），使用共享环境。

```powershell
uv pip install --python ..\\.venv\\Scripts\\python.exe -r requirements-advanced.txt
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/prepare_temporal_history.py
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/prepare_temporal_history.py --full
..\\.venv\\Scripts\\python.exe run_advanced.py --trials
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/prepare_rating_features.py
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/calibrate_advanced.py --name calibration_v2_d6 --depth 6 --iterations 700
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/calibrate_advanced.py --name calibration_v2_d8 --depth 8 --iterations 1200
..\\.venv\\Scripts\\python.exe run_advanced.py --calibration-ablations
```

`freeze_advanced.py`只允许在新实验目录冻结一次，已有冻结结果会拒绝重开，避免按审计调参。新一次复现应先保留原产物、使用新工作副本。

```powershell
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/freeze_advanced.py --candidates calibration_v2_d6 calibration_v2_d8 calibration_v2_d8_no_semantics --ablations calibration_v2_d8_no_history calibration_v2_d8_no_neural calibration_v2_d8_seed2 mf64_r003 mf64_r03 mf64_r08 mf128_r04 svdpp32_r03 memory32_time_r03 memory32_notime_r03 mf64_time_r03
..\\.venv\\Scripts\\python.exe run_advanced.py --refit mf64_r003 mf64_r03 mf64_r08 mf128_r04 svdpp32_r03 memory32_time_r03 memory32_notime_r03 mf64_time_r03
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/prepare_rating_features.py --full
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/predict_advanced.py
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/test_advanced.py
..\\.venv\\Scripts\\python.exe 06_integration_ablation/scripts/verify_advanced.py
..\\.venv\\Scripts\\python.exe docs/create_advanced_figures.py
..\\.venv\\Scripts\\python.exe docs/summarize_advanced.py
```

'''
    body=body[:idx]+block+body[idx:]
    body=body.replace('## 当前方案：冻结 Qwen3 语义、Transformer、MF 与 ALS','## 上一轮归档：冻结 Qwen3 语义、Transformer、MF 与 ALS')
    body=body.replace('当前预测文件 `data/raw/hybrid/test_pairs_predictions_transformer.csv`','上一轮预测文件 `data/raw/hybrid/test_pairs_predictions_transformer.csv`')
    stage.write_text(body,encoding='utf-8')
    stages=json.loads((ROOT/'stages.json').read_text(encoding='utf-8'));six=stages[-1]
    six['goal']=f'混合数据候选历史记忆、时间MF与OOF残差校正；本轮审计RMSE{new["audit_rmse"]:.6f}，外层合并{new["overall_rmse"]:.6f}；消融与100000条预测完成，测试无标签。'
    six['advanced_enhancement']={'date':'2026-11-23','config':'advanced_config.json','audit_rmse':new['audit_rmse'],'overall_oof_audit_rmse':new['overall_rmse'],'selected':selected,'method':'../docs/advanced_method.md','verification':'logs/advanced/verification.json'}
    for path in ['advanced_config.json','logs/advanced/audit_evaluation.json','logs/advanced/verification.json','logs/advanced/prediction_manifest.json','scripts/advanced_rating.py','scripts/calibrate_advanced.py']:
        if path not in six['acceptance_files']:six['acceptance_files'].append(path)
    (ROOT/'stages.json').write_text(json.dumps(stages,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'audit_rmse':new['audit_rmse'],'overall_oof_audit_rmse':new['overall_rmse'],'relative_audit_gain_percent':rel},indent=2))


if __name__=='__main__':main()
