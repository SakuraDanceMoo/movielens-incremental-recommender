"""Course-paper figures sourced exclusively from saved stage-six experiments."""
from __future__ import annotations
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'.venv/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOGS=ROOT/'06_integration_ablation/logs/advanced';OUT=ROOT/'docs/reports/assets/advanced'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False,'font.size':13,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
TEAL='#007F83';BLUE='#4177A6';GREY='#8C9AA7';NAVY='#12263A';LIGHT='#EAF1F5'
SOURCES={}
def read(name):return json.loads((LOGS/name).read_text(encoding='utf-8'))
def save(fig,name,sources):
    OUT.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(OUT/name,dpi=220,bbox_inches='tight');plt.close(fig);SOURCES[name]=sources
def box(ax,x,y,w,h,text,color=LIGHT):
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor=color,edgecolor='none',transform=ax.transAxes))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color='white' if color==TEAL else NAVY,fontsize=12.5,transform=ax.transAxes)
def arrow(ax,p,q):ax.annotate('',xy=q,xytext=p,xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})


def main():
    evaluation=read('audit_evaluation.json');selected=evaluation['selected'];sel=read(f'result_{selected}.json')
    rows={r['name']:r for r in evaluation['rows']}
    fig,ax=plt.subplots(figsize=(11,5));ax.axis('off')
    box(ax,.01,.68,.29,.22,'拟合评分 + 查询用户 / 电影\n严格早于查询的最近16条历史')
    box(ax,.38,.70,.25,.20,'候选条件4头交叉注意力\n评分 / 位置 / 历史记忆')
    box(ax,.38,.35,.25,.22,'正则MF / 时间偏置 / SVD++\n既有Qwen Transformer与ALS')
    box(ax,.01,.13,.29,.22,'用户 / 电影统计\n类型偏好与历史相似度')
    box(ax,.74,.35,.25,.35,f'开发用户五折残差校正\n{sel["feature_count"]}个入模特征\n旧融合评分 + 预测残差\n裁剪至0.5至5.0',TEAL)
    for p,q in [((.3,.78),(.38,.8)),((.3,.73),(.38,.51)),((.3,.24),(.74,.41)),((.63,.8),(.74,.64)),((.63,.46),(.74,.52))]:arrow(ax,p,q)
    ax.text(.5,.01,'结构示意；借鉴近期候选条件记忆思想，属于评分回归改编',ha='center',transform=ax.transAxes,color=GREY,fontsize=12)
    ax.set_title('阶段六　历史记忆、协同模型与残差校正',fontsize=16)
    save(fig,'architecture.png',['06_integration_ablation/scripts/advanced_rating.py','06_integration_ablation/scripts/prepare_rating_features.py',f'06_integration_ablation/logs/advanced/result_{selected}.json'])

    fig,ax=plt.subplots(figsize=(11,4.9));ax.axis('off')
    box(ax,.02,.69,.28,.21,'原训练表 31,900,204条\n逐用户最新评分留出')
    box(ax,.38,.72,.25,.18,'拟合 31,513,222条\n训练所有基础推荐模型')
    box(ax,.38,.32,.25,.24,'开发 309,513位用户\n五折校正 / 折外选结构\n不使用本条标签拟合校正器')
    box(ax,.74,.32,.24,.24,'审计 77,469位用户\n本轮固定方案后检查\n五个校正器预测取均值',TEAL)
    arrow(ax,(.3,.79),(.38,.81));arrow(ax,(.16,.69),(.38,.5));arrow(ax,(.63,.45),(.74,.45))
    ax.text(.5,.12,'基础模型始终使用拟合集；开发标签仅训练第二层校正器',ha='center',transform=ax.transAxes)
    ax.text(.5,.02,'审计仅对本轮保留；旧项目模型曾用整个留出集调参，仍需全新隐藏测试',ha='center',transform=ax.transAxes,color=GREY,fontsize=11.5)
    ax.set_title('阶段六　评估协议与标签使用范围',fontsize=16)
    save(fig,'evaluation_protocol.png',['06_integration_ablation/scripts/calibrate_advanced.py','06_integration_ablation/logs/advanced/frozen_selection.json'])

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    variants=[('mf64_r003','MF λ=0.003'),('mf64_r03','MF λ=0.03'),('mf64_time_r03','时间MF λ=0.03'),('memory32_time_r03','候选历史注意力')]
    for name,label in variants:
        h=read(f'history_{name}_validation.json');ep=[v['epoch'] for v in h]
        axes[0].plot(ep,[v['train_mse'] for v in h],'o-',label=label,markersize=4)
        axes[1].plot(ep,[v['development']['rmse'] for v in h],'o-',label=label,markersize=4)
    for ax in axes:ax.set_xlabel('完整拟合集遍历轮次');ax.grid(alpha=.15);ax.set_xticks(range(1,8))
    axes[0].set_title('真实训练损失');axes[0].set_ylabel('批次累计 MSE');axes[0].legend(fontsize=9)
    axes[1].set_title('开发误差与过拟合');axes[1].set_ylabel('开发 RMSE');axes[1].legend(fontsize=9)
    save(fig,'training_curves.png',[f'06_integration_ablation/logs/advanced/history_{n}_validation.json' for n,_ in variants])

    names=['calibration_v2_d8','calibration_v2_d8_no_history','calibration_v2_d8_no_neural','calibration_v2_d8_no_semantics']
    labels=['完整75特征','移除历史特征','移除本轮神经特征','移除额外语义特征']
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.5))
    for ax,field,title in zip(axes,['development_rmse','audit_rmse'],['开发用户五折折外','固定方案后审计']):
        vals=[rows[n][field] for n in names];ypos=np.arange(len(vals))
        ax.barh(ypos,vals,color=[TEAL,BLUE,GREY,TEAL],height=.6);ax.set_yticks(ypos,labels if field=='development_rmse' else ['']*4);ax.invert_yaxis()
        ax.set_xlim(.85,max(vals)+.007);ax.set_xlabel('RMSE（横轴局部放大）');ax.set_title(title);ax.grid(axis='x',alpha=.15)
        for j,v in enumerate(vals):ax.text(v+.00035,j,f'{v:.6f}',va='center',fontsize=11)
    save(fig,'feature_ablation.png',['06_integration_ablation/logs/advanced/advanced_comparison.csv'])

    fig,ax=plt.subplots(figsize=(10.5,4.3))
    compare=['original_als','previous_transformer_ensemble',selected];labels=['原 ALS','上一轮 Qwen / MF / ALS','本轮选定方案']
    vals=[rows[n]['audit_rmse'] for n in compare];x=np.arange(3)
    ax.bar(x,vals,color=[GREY,BLUE,TEAL],width=.55)
    for xx,v in zip(x,vals):ax.text(xx,v+.002,f'{v:.6f}',ha='center',fontweight='bold')
    ax.set_xticks(x,labels);ax.set_ylim(.82,max(vals)+.02);ax.set_ylabel('审计 RMSE（纵轴局部放大）');ax.set_title('同一77,469位审计用户的配对比较');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'audit_comparison.png',['06_integration_ablation/logs/advanced/audit_evaluation.json'])
    (OUT/'figure_sources.json').write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figures':len(SOURCES),'directory':str(OUT)},ensure_ascii=False))


if __name__=='__main__':main()
