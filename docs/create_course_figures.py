"""Rebuild course-report figures from logged measurements, without rerunning training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'.venv/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=ROOT/'docs/reports/assets'
TEAL='#007F83';NAVY='#12263A';BLUE='#4177A6';GREY='#8898A7';LIGHT='#EAF1F5'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold',
                     'savefig.facecolor':'white'})
SOURCES={}


def read(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8'))
def save(fig,name,sources):
    OUT.mkdir(parents=True,exist_ok=True)
    # Figures are printed at about 6 inches in an A4 paper, smaller than their
    # authoring canvas. Enlarge labels before final layout for readable print.
    for item in fig.findobj(match=matplotlib.text.Text):
        item.set_fontsize(item.get_fontsize()*1.35)
    fig.tight_layout()
    fig.savefig(OUT/name,dpi=220,bbox_inches='tight')
    plt.close(fig);SOURCES[name]=sources
def bars(ax,labels,values,colors=None):
    positions=np.arange(len(labels))
    ax.barh(positions,values,color=colors or BLUE,height=.6)
    ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set_xlim(0,max(values)*1.18)
    ax.set_xlabel('验证 RMSE，越低越好');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for j,v in enumerate(values):ax.text(v+.012,j,f'{v:.6f}',va='center',fontsize=10)


def base_figures():
    audit=read('01_data_baseline/logs/hybrid_data_audit.json')
    old=read('06_integration_ablation/logs/rating_prediction_metrics.json')
    sim=read('02_similarity_recall/logs/hybrid_rating_metrics.json')
    stream=read('04_streaming_incremental/logs/hybrid_rating_metrics.json')
    graph=read('05_graph_recommendation/logs/hybrid_rating_metrics.json')
    fig,ax=plt.subplots(1,2,figsize=(10.8,4.2),gridspec_kw={'width_ratios':[1,1.2]})
    vals=[audit['ratings']['movielens32m_rows'],audit['ratings']['synthetic_rows']]
    ax[0].pie(vals,labels=['MovieLens\n15,950,149 条','合成数据\n15,950,055 条'],colors=[TEAL,BLUE],autopct='%1.2f%%',
              startangle=90,wedgeprops={'width':.5,'edgecolor':'white'},textprops={'fontsize':10})
    ax[0].set_title('训练评分来源')
    ax[1].axis('off')
    lines=[('训练评分','31,900,204'),('用户 / 电影','396,060 / 142,609'),('拟合评分','31,513,222'),
           ('时间留出验证','386,982'),('测试配对（无标签）','100,000'),('允许评分范围','0.5–5.0')]
    for idx,(label,value) in enumerate(lines):
        y=.92-idx*.16
        ax[1].text(.02,y,label,transform=ax[1].transAxes,color=NAVY)
        ax[1].text(.98,y,value,transform=ax[1].transAxes,ha='right',fontweight='bold',color=TEAL)
    fig.tight_layout();save(fig,'stage1_hybrid_data.png',['01_data_baseline/logs/hybrid_data_audit.json','06_integration_ablation/logs/rating_prediction_metrics.json'])

    fig,ax=plt.subplots(figsize=(9.5,3.7))
    base=[old['validation']['model_trials'][0]['validation'][k]['global_mean_rmse'] for k in ['overall','movielens32m','synthetic']]
    vals=[sim['validation']['rmse'],sim['validation']['movielens32m']['rmse'],sim['validation']['synthetic']['rmse']]
    x=np.arange(3);w=.3
    ax.bar(x-w/2,base,w,label='全局均值',color=GREY);ax.bar(x+w/2,vals,w,label='类型相似度',color=TEAL)
    for xx,vv in [(x-w/2,base),(x+w/2,vals)]:
        for a,b in zip(xx,vv):ax.text(a,b+.013,f'{b:.4f}',ha='center',fontsize=10)
    ax.set_xticks(x,['全部验证用户','MovieLens 用户','合成用户']);ax.set_ylabel('验证 RMSE');ax.set_ylim(0,1.27);ax.legend(ncol=2,loc='lower right')
    ax.set_title('阶段二　相似度评分回归在不同来源上的表现');fig.tight_layout()
    save(fig,'stage2_similarity_rmse.png',['02_similarity_recall/logs/hybrid_rating_metrics.json','06_integration_ablation/logs/rating_prediction_metrics.json'])

    fig,ax=plt.subplots(figsize=(9.5,3.7))
    trials=old['validation']['model_trials']
    for k,(name,color) in enumerate([('overall',TEAL),('movielens32m',BLUE),('synthetic',GREY)]):
        values=[t['validation'][name]['rmse_clipped'] for t in trials]
        ax.plot([16,32],values,'o-',color=color,label=['全部','MovieLens','合成'][k],linewidth=2)
        for x,y in zip([16,32],values):ax.annotate(f'{y:.6f}',(x,y),xytext=(8,7 if k!=1 else -17),textcoords='offset points',fontsize=9)
    ax.set_xticks([16,32]);ax.set_xlim(13,38);ax.set_xlabel('ALS 隐向量维度 rank');ax.set_ylabel('验证 RMSE');ax.grid(alpha=.15);ax.legend()
    ax.set_title('阶段三　显式 ALS 参数对照（相同划分、8轮迭代、正则0.1）');fig.tight_layout()
    save(fig,'stage3_als_rank.png',['06_integration_ablation/logs/rating_prediction_metrics.json'])

    fig,ax=plt.subplots(1,2,figsize=(10.8,4.4),gridspec_kw={'width_ratios':[1.3,1]})
    ax[0].axis('off');ys=[.84,.54,.24]
    for y,txt in zip(ys,['按事件时间排序评分','更新全局 / 用户 / 电影均值','仅用严格早于查询时刻的状态']):
        ax[0].text(.5,y,txt,ha='center',va='center',bbox={'boxstyle':'square,pad=.7','facecolor':LIGHT,'edgecolor':'none'},transform=ax[0].transAxes)
    for y in [.75,.45]:ax[0].annotate('',xy=(.5,y-.1),xytext=(.5,y),xycoords='axes fraction',arrowprops={'arrowstyle':'->','color':TEAL,'lw':2})
    ax[0].text(.5,.04,'方法示意；配置权重为 0.50 / 0.25 / 0.25',ha='center',fontsize=10,transform=ax[0].transAxes,color=GREY)
    ax[0].set_title('事件时间增量状态')
    bars(ax[1],['全局均值','增量均值'],[old['validation']['global_mean_baseline_rmse'],stream['validation']['rmse']],[GREY,TEAL])
    ax[1].set_title('实际验证结果');fig.tight_layout()
    save(fig,'stage4_event_time.png',['04_streaming_incremental/scripts/hybrid_rating_regression.py','04_streaming_incremental/logs/hybrid_rating_metrics.json'])

    fig,ax=plt.subplots(figsize=(9.5,3.9))
    x=np.arange(3);w=.24
    groups=[[sim['validation']['rmse'],sim['validation']['movielens32m']['rmse'],sim['validation']['synthetic']['rmse']],
            [graph['validation']['rmse'],graph['validation']['movielens32m']['rmse'],graph['validation']['synthetic']['rmse']],
            [trials[0]['validation'][k]['rmse_clipped'] for k in ['overall','movielens32m','synthetic']]]
    for idx,(label,values,color) in enumerate(zip(['类型相似度','度数归一化图传播','ALS rank16'],groups,[BLUE,GREY,TEAL])):
        positions=x+(idx-1)*w;ax.bar(positions,values,w,label=label,color=color)
        for a,b in zip(positions,values):ax.text(a,b+.015,f'{b:.4f}',ha='center',fontsize=9)
    ax.set_xticks(x,['全部验证用户','MovieLens 用户','合成用户']);ax.set_ylim(0,1.28);ax.set_ylabel('验证 RMSE');ax.legend(loc='lower right')
    ax.set_title('阶段五　图传播未优于相似度与 ALS，保留实测负结果');fig.tight_layout()
    save(fig,'stage5_graph_result.png',['05_graph_recommendation/logs/hybrid_rating_metrics.json','02_similarity_recall/logs/hybrid_rating_metrics.json','06_integration_ablation/logs/rating_prediction_metrics.json'])


def transformer_figures():
    selection=read('06_integration_ablation/logs/transformer/selection.json')
    ensemble=read('06_integration_ablation/logs/transformer/ensemble_selection.json')
    fig,ax=plt.subplots(figsize=(11,5));ax.axis('off')
    items=[(.02,.70,.28,.16,'电影标题与类型\n冻结 Qwen3-Embedding-0.6B'),(.38,.70,.25,.16,'128维语义向量\n类型与年份特征'),
           (.02,.25,.28,.16,'用户 / 电影 ID\n可选 ALS 因子初始化'),(.38,.25,.25,.16,'32维 ID 向量与 MF\n两分支独立训练'),
           (.73,.55,.25,.24,'四类特征 token\n4头自注意力 + SwiGLU\n监督评分回归'),
           (.73,.10,.25,.20,'MF / ALS / Transformer\n权重 0.55 / 0.20 / 0.25\n评分裁剪到0.5至5.0'),(.38,.05,.25,.10,'原 Spark ALS 评分')]
    for x,y,w,h,t in items:
        ax.add_patch(plt.Rectangle((x,y),w,h,transform=ax.transAxes,facecolor=LIGHT if x<.7 else TEAL,edgecolor='none'))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',transform=ax.transAxes,color=NAVY if x<.7 else 'white',fontsize=11)
    for p,q in [((.3,.78),(.38,.78)),((.3,.33),(.38,.33)),((.63,.78),(.73,.72)),((.63,.33),(.73,.60)),((.855,.55),(.855,.30)),((.63,.30),(.73,.24)),((.63,.10),(.73,.16))]:
        ax.annotate('',xy=q,xytext=p,xycoords='axes fraction',arrowprops={'arrowstyle':'->','lw':2,'color':TEAL})
    ax.text(.02,.02,'结构示意；冻结文本编码器，三个评分分支按验证结果融合。',fontsize=10,color=GREY,transform=ax.transAxes)
    fig.tight_layout();save(fig,'stage6_transformer_architecture.png',['06_integration_ablation/scripts/transformer_regression.py','06_integration_ablation/scripts/encode_movie_semantics.py'])

    rows=selection['ablation_summary']+[{'variant':'final_ensemble','mean_rmse':ensemble['two_seed_mean_rmse'],'std_rmse':ensemble['two_seed_std_rmse']}];labels={'full':'ALS初始化 + Qwen + Transformer','no_semantics':'ALS初始化，移除 Qwen 语义','no_attention':'ALS初始化，自注意力换为 MLP','no_als_init':'随机初始化 + Qwen + Transformer','mf_only':'ALS初始化，仅因子微调',
        'random_no_semantics':'随机初始化，移除 Qwen 语义','random_no_attention':'随机初始化，自注意力换为 MLP','random_mf_only':'随机初始化，仅因子微调','final_ensemble':'最终三分支融合'}
    fig,ax=plt.subplots(figsize=(10.5,6.3));y=np.arange(len(rows))
    for pos,r in zip(y,rows):
        ax.errorbar(r['mean_rmse'],pos,xerr=r['std_rmse'],fmt='o',capsize=5,color=TEAL if r['variant']==selection['variant'] else BLUE,markersize=7)
        ax.annotate(f"{r['mean_rmse']:.6f} ± {r['std_rmse']:.6f}",(r['mean_rmse'],pos),xytext=(8,-17),textcoords='offset points',fontsize=9)
    ax.set_yticks(y,[labels[r['variant']] for r in rows]);ax.invert_yaxis();ax.set_xlabel('验证 RMSE，两随机种子均值 ± 样本标准差');ax.grid(axis='x',alpha=.2)
    lo=min(r['mean_rmse']-r['std_rmse'] for r in rows);hi=max(r['mean_rmse']+r['std_rmse'] for r in rows)
    ax.set_xlim(lo-.001,hi+.006);ax.set_title('阶段六　单组件消融，固定数据、训练预算和融合候选');fig.tight_layout()
    save(fig,'stage6_transformer_ablation.png',['06_integration_ablation/logs/transformer/ablation.csv','06_integration_ablation/logs/transformer/ensemble_selection.json'])

    fig,ax=plt.subplots(1,2,figsize=(10.8,4))
    for variant,color in [('no_als_init',TEAL),('random_no_semantics',BLUE),('random_mf_only',GREY)]:
        history=read(f'06_integration_ablation/logs/transformer/history_{variant}_validation_seed20261123.json')
        epochs=[r['epoch'] for r in history]
        ax[0].plot(epochs,[r['train_mse'] for r in history],'o-',color=color,label=labels[variant])
        ax[1].plot(epochs,[r['rmse'] for r in history],'o-',color=color,label=labels[variant])
    ax[0].set_ylabel('训练 MSE');ax[1].set_ylabel('融合后验证 RMSE')
    for a in ax:a.set_xlabel('训练轮次');a.set_xticks([1,2,3,4]);a.grid(alpha=.2)
    final_history=read('06_integration_ablation/logs/transformer/history_no_als_init_validation_seed20261123.json')
    ax[1].plot([r['epoch'] for r in final_history],[r['raw_neural_rmse'] for r in final_history],'--',color=TEAL,alpha=.7,label='最终结构的神经分支（融合前）')
    ax[1].axhline(selection['baseline_rmse'],color=NAVY,linestyle=':',label='原 ALS')
    ax[0].legend(fontsize=8);ax[1].legend(fontsize=8);ax[0].set_title('真实训练损失');ax[1].set_title('验证选择曲线，非隐藏测试集');fig.tight_layout()
    save(fig,'stage6_training_curves.png',['06_integration_ablation/logs/transformer/history_no_als_init_validation_seed20261123.json','06_integration_ablation/logs/transformer/history_random_no_semantics_validation_seed20261123.json','06_integration_ablation/logs/transformer/history_random_mf_only_validation_seed20261123.json'])

    fig,ax=plt.subplots(figsize=(10,4.5))
    vals=[1.0675994768057624,1.0473233484558542,1.0058308529922668,1.0630981509111395,selection['baseline_rmse'],ensemble['mf_als_fused_control_rmse'],ensemble['validation_rmse']]
    bars(ax,['全局均值','类型相似度','事件时间增量','图传播','原 ALS rank16','新 MF 与 ALS 融合','最终三分支融合'],vals,[GREY,BLUE,BLUE,BLUE,NAVY,BLUE,TEAL]);ax.set_title('统一 386,982 条验证评分上的方法对比');fig.tight_layout()
    save(fig,'stage6_overall_comparison.png',['06_integration_ablation/logs/hybrid_rating_comparison.csv','06_integration_ablation/logs/transformer/ensemble_selection.json'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base-only',action='store_true');args=parser.parse_args()
    base_figures()
    if not args.base_only:transformer_figures()
    (OUT/'figure_sources.json').write_text(json.dumps(SOURCES,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'figures':len(SOURCES),'directory':str(OUT)},ensure_ascii=False))
