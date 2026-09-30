from pathlib import Path
import json, hashlib, shutil, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve, average_precision_score

P=Path(__file__).resolve().parents[1]; W=P/'outputs/figure_build'; A=P/'analysis'
R=A/'EUOS_locked_fusion_evaluation_2026-09-21_v2'; F=A/'Cross_context_fair_increment_followup_2026-09-20_v1'; C=A/'Remaining_evidence_closure_2026-09-20_v1'
D=P/'outputs'; FIG=D/'Figures'; SD=W/'source_data'
for d in [D,FIG,SD,W/'previews',W/'qa']: d.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False,'legend.fontsize':7})
BLUE='#326E9B'; TEAL='#27847B'; ORANGE='#BE713B'; PURPLE='#80619A'; GRAY='#7F8990'; LIGHT='#E6E9EC'
COL={'B':BLUE,'M':TEAL,'F':ORANGE,'similarity_natural':PURPLE}
NAMES={'B':'Structure + count','M':'Morphology','F':'Selected fusion','similarity_natural':'Adapted similarity comparator'}
registry=[]; inputs={}; legends=[]
def read(root,name):
    p=root/name; inputs[str(p.relative_to(P))]=hashlib.sha256(p.read_bytes()).hexdigest(); return pd.read_csv(p,keep_default_na=False,na_values=[''])
def source(fig,panel,df,origin,unit,selection='All rows in the stated comparison; no significance filtering.'):
    f=SD/f'{fig}_{panel}.csv'; df.to_csv(f,index=False)
    registry.append(dict(figure=fig,panel=panel,source_data=str(f),input=origin,rows=len(df),unit=unit,selection=selection))
def title(ax,letter,s):
    heading=ax.set_title(s,loc='left',pad=12,y=1.0)
    ax.text(-.045 if not ax.axison else -.14,1.0,letter,transform=heading.get_transform(),fontweight='bold',fontsize=10,va='baseline')
def save(fig,name,caption,title_text):
    fig.savefig(FIG/f'{name}.pdf')
    fig.savefig(FIG/f'{name}.svg')
    fig.savefig(FIG/f'{name}.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(W/'previews'/f'{name}.png',dpi=300)
    plt.close(fig); legends.append(dict(id=name,title=title_text,legend=caption+' Source data are provided in the accompanying archive.'))
def gain(df):
    z=df.copy(); z['gain']=-z.mean_difference; z['low']=-z.ci95_high; z['high']=-z.ci95_low; return z
def two(h=3.7,left=.13,wspace=.6):
    fig,ax=plt.subplots(1,2,figsize=(7.0866,h)); fig.subplots_adjust(left=left,right=.975,bottom=.20,top=.85,wspace=wspace); return fig,ax
def intervals(ax,labels,p,lo,hi,color=BLUE):
    p,lo,hi=map(np.asarray,[p,lo,hi]); y=np.arange(len(p))
    ax.axvline(0,color=GRAY,lw=.65,ls='--'); ax.errorbar(p,y,xerr=[p-lo,hi-p],fmt='o',ms=3.5,color=color,capsize=2,lw=.85)
    ax.set_yticks(y,labels); ax.invert_yaxis(); ax.margins(y=.22)

# Figure 1: the design, not an illustration of an unmeasured mechanism.
u=read(R,'unit_partition_labels.csv'); q=read(R,'endpoint_qualification.csv')
labelled=u.loc[u[q.assay.tolist()].notna().any(axis=1)]; counts=labelled.partition.value_counts()
assert [counts[k] for k in ['development','strict_holdout','new_entity_seen_scaffold']]==[353,145,78]
fig,ax=plt.subplots(2,1,figsize=(7.0866,5.5),gridspec_kw={'height_ratios':[1.15,1]}); fig.subplots_adjust(left=.07,right=.975,bottom=.08,top=.90,hspace=.60)
for a in ax: a.set_xlim(0,1); a.set_ylim(0,1); a.axis('off')
def box(a,x,y,w,h,s,color=BLUE,fill='#F3F6F8',size=8):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',facecolor=fill,edgecolor=color,lw=.8)); a.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=size,linespacing=1.4)
def arrow(a,xy1,xy2): a.annotate('',xy=xy2,xytext=xy1,arrowprops={'arrowstyle':'->','color':GRAY,'lw':.8})
title(ax[0],'a','Compound allocation')
box(ax[0],.30,.76,.4,.18,'576 labeled parent compounds')
xs=[.015,.352,.689]; texts=['Development\n353 compounds','New entity / new scaffold\n145 compounds','New entity / seen scaffold\n78 compounds']
for x,s in zip(xs,texts): box(ax[0],x,.33,.285,.23,s); arrow(ax[0],(.5,.75),(x+.1425,.58))
ax[0].text(.50,.095,'Qualified strict test: SR-ARE 85; SR-MMP 95\n117 distinct compounds; 114 scaffolds',ha='center',va='center',fontsize=8)
arrow(ax[0],(.4945,.31),(.4945,.19))
title(ax[1],'b','Development and transfer comparisons')
box(ax[1],.015,.47,.30,.40,'Model development\nFMP · IMTM · MEDINA\nHepG2',color=GRAY)
box(ax[1],.385,.47,.265,.40,'External evaluation\nUSC\nHepG2',color=BLUE)
box(ax[1],.72,.47,.265,.40,'Paired context test\nFMP\nHepG2 → U2OS',color=TEAL)
arrow(ax[1],(.33,.67),(.37,.67)); arrow(ax[1],(.665,.67),(.705,.67))
ax[1].text(.5,.18,'Identical compound identities and activity labels in the paired test\nCellular background and acquisition conditions change together',ha='center',va='center',fontsize=8)
source('Figure_1','a',counts.rename('n').reset_index(),'unit_partition_labels.csv','Standardized parent compound')
source('Figure_1','b',pd.DataFrame([['Development','FMP; IMTM; MEDINA','HepG2'],['External','USC','HepG2'],['Paired','FMP','HepG2; U2OS']],columns=['role','laboratory','cell_background']),'implementation_contract.md','Experimental context')
save(fig,'Figure_1','a, Allocation by prior chemical history. Of 576 labeled parent compounds, 573 had profiles at all four HepG2 laboratories; endpoint qualification additionally required a nonmissing activity label. SR-ARE and SR-MMP strict tests overlap and are not 180 independent compounds. b, Models are developed in three HepG2 laboratories and evaluated in an excluded laboratory. The paired comparison changes the cellular input for the same strict held-out compounds; it does not isolate cell lineage from acquisition conditions. Arrows denote analysis comparisons, not chronological sample collection. Counts have no uncertainty intervals.','Study design and evaluation populations')

tc=read(R,'external_task_comparisons.csv'); mc=read(R,'external_macro_comparisons.csv'); met=read(R,'external_task_metrics.csv'); pred=read(R,'scored_predictions.csv')
strict=pred.query("partition=='strict_holdout' and family=='lr'")
external=met.query("partition=='strict_holdout' and family=='lr' and site=='USC_HepG2'")
fig,axs=plt.subplots(2,2,figsize=(7.0866,6.7)); fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.93,hspace=.85,wspace=.57)
a,b,c,d=axs.ravel(); assays=['SR-ARE','SR-MMP']
for i,assay in enumerate(assays):
    z=external.query('assay==@assay').set_index('model').loc[['B','M','F']]
    a.plot(np.arange(3),z.brier,color=GRAY,lw=.8,ls=['-','--'][i],zorder=1)
    for j,(model,row) in enumerate(z.iterrows()): a.scatter(j,row.brier,c=COL[model],marker=['o','s'][i],s=28,zorder=3)
    a.text(2.10,z.loc['F','brier'],assay,fontsize=7,va='center')
a.set_xticks([0,1,2],['Structure\n+ count','Morphology','Selected\nfusion']); a.set_xlim(-.25,3.02); a.set_ylim(.10,.23); a.set_ylabel('Brier score (lower is better)'); title(a,'a','Absolute prediction error')
rows=[]
for j,family in enumerate(['lr','rf']):
    z=gain(tc.query("partition=='strict_holdout' and site=='USC_HepG2' and family==@family and model=='F' and reference=='B'"))
    zz=gain(mc.query("partition=='strict_holdout' and site=='USC_HepG2' and family==@family and model=='F' and reference=='B'")).copy(); zz['assay']='Two-endpoint mean'; z=pd.concat([z,zz]); rows.append(z)
    for i,assay in enumerate(assays+['Two-endpoint mean']):
        r=z[z.assay==assay].iloc[0]; yy=i+(j-.5)*.23
        b.errorbar(r.gain,yy,xerr=[[r.gain-r.low],[r.high-r.gain]],fmt=['o','s'][j],color=[ORANGE,GRAY][j],ms=3.5,capsize=2,lw=.85,label=['Logistic regression','Random forest'][j] if i==0 else None)
b.axvline(0,color=GRAY,lw=.7,ls='--'); b.set_yticks(range(3),['SR-ARE','SR-MMP','Two-endpoint\nmean']); b.invert_yaxis(); b.margins(y=.2); b.set_xlim(-.02,.09); b.set_xlabel('Brier gain (baseline − fusion)'); b.legend(loc='upper left',bbox_to_anchor=(0,-.25)); title(b,'b','Increment and uncertainty')
curves=[]
for a,assay,letter in zip([c,d],assays,['c','d']):
    z=strict.query("site=='USC_HepG2' and assay==@assay")
    for model in ['B','M','F','similarity_natural']:
        pp,rr,tt=precision_recall_curve(z.y,z[model]); a.step(rr,pp,where='post',color=COL[model],lw=1.1,label=f'{NAMES[model]} ({average_precision_score(z.y,z[model]):.3f})')
        curves.append(pd.DataFrame({'assay':assay,'model':model,'recall':rr,'precision':pp}))
    a.axhline(z.y.mean(),c=GRAY,lw=.7,ls=':'); a.set(xlim=(0,1),ylim=(0,1.03),xlabel='Recall',ylabel='Precision'); a.legend(loc='upper left',bbox_to_anchor=(0,-.30),fontsize=7,handlelength=1.4); title(a,letter,assay+' ranking')
source('Figure_2','a',external[external.model.isin(['B','M','F'])],'external_task_metrics.csv','Endpoint and model')
source('Figure_2','b',pd.concat(rows),'external_task_comparisons.csv; external_macro_comparisons.csv','Fixed-model paired scaffold-bootstrap estimate')
source('Figure_2','cd',pd.concat(curves),'scored_predictions.csv','Precision–recall threshold; descriptive curve')
source('Figure_2','predictions',strict.query("site=='USC_HepG2'")[['parent_key','scaffold','assay','y','B','M','F','similarity_natural']],'scored_predictions.csv','Held-out compound–endpoint prediction')
save(fig,'Figure_2','a, Observed Brier scores from logistic-regression models using structure plus count, morphology alone or selected fusion in USC-HepG2, with circles for SR-ARE and squares for SR-MMP. Connecting lines identify an endpoint across categorical inputs, not a continuous trajectory. b, Brier gains for logistic regression and random forest. Points and bars show estimates and percentile 95% intervals from 5,000 paired scaffold bootstrap resamples; positive favors fusion. c,d, Precision–recall curves from logistic-regression predictions. Legend values are average precision; dotted lines mark the active fraction. Curves are descriptive, without confidence bands. SR-ARE has 85 compounds/85 singleton scaffolds; SR-MMP has 95 compounds/92 scaffolds; their union has 117 compounds/114 scaffolds. Joint resampling preserves endpoint overlap and does not refit models. Intervals are not multiplicity-adjusted. The adapted similarity comparator is not an exact reproduction of the original benchmark.','Prediction gains and screening trade-offs')

pair=gain(tc.query("partition=='strict_holdout' and family=='lr' and model=='F' and reference=='B' and site in ['FMP_HepG2','FMP_U2OS']")); contrast=read(R,'paired_cell_context_uncertainty.csv').query("family=='lr'")
fig,ax=two(3.65,left=.11,wspace=.6); pp=[]
for assay,color,marker in zip(assays,[ORANGE,TEAL],['o','s']):
    z=pair[pair.assay==assay].set_index('site').loc[['FMP_HepG2','FMP_U2OS']]
    ax[0].errorbar([0,1],z.gain,yerr=[z.gain-z.low,z.high-z.gain],color=color,fmt=marker+'-',lw=1.15,capsize=3,ms=4,label=assay)
    z=strict.query("assay==@assay and site in ['FMP_HepG2','FMP_U2OS']").copy(); z['compound_gain']=(z.B-z.y)**2-(z.F-z.y)**2
    piv=z.pivot(index=['parent_key','scaffold'],columns='site',values='compound_gain').reset_index(); piv['assay']=assay; pp.append(piv)
    ax[1].scatter(piv.FMP_HepG2,piv.FMP_U2OS,s=12,c=color,marker=marker,alpha=.6,label=assay,linewidths=0)
ax[0].axhline(0,c=GRAY,lw=.7,ls='--'); ax[0].set_xticks([0,1],['FMP-HepG2','FMP-U2OS']); ax[0].set_xlim(-.25,1.25); ax[0].set_ylabel('Mean Brier gain'); ax[0].legend(loc='upper right'); title(ax[0],'a','Change in incremental value')
allp=pd.concat(pp); lim=max(abs(allp[['FMP_HepG2','FMP_U2OS']].to_numpy().ravel()))*1.07
ax[1].plot([-lim,lim],[-lim,lim],c=GRAY,lw=.7,ls=':'); ax[1].axhline(0,c=GRAY,lw=.65,ls='--'); ax[1].axvline(0,c=GRAY,lw=.65,ls='--'); ax[1].set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='Compound gain in HepG2',ylabel='Compound gain in U2OS'); title(ax[1],'b','Paired compound-level changes')
source('Figure_3','a',pair,'external_task_comparisons.csv','Endpoint-context gain'); source('Figure_3','b',allp,'scored_predictions.csv','Paired compound–endpoint gain'); source('Figure_3','paired_contrast',contrast,'paired_cell_context_uncertainty.csv','Paired context difference')
save(fig,'Figure_3','a, Logistic-regression mean Brier gain in two FMP experimental contexts for identical held-out compounds. Bars show percentile 95% intervals from 5,000 scaffold resamples. The paired U2OS-minus-HepG2 changes are −0.0534 (95% CI, −0.1068 to −0.0018) for SR-ARE and −0.0357 (−0.0633 to −0.0100) for SR-MMP. b, Each point shows the same compound–endpoint pair in both contexts, using (baseline probability − label)² − (fusion probability − label)². Points below the identity line have lower gains in U2OS. All 85 SR-ARE and 95 SR-MMP pairs are shown, representing 117 distinct compounds overall; endpoint overlap does not create independent replicates. Zero lines indicate equal baseline and fusion error. Biological background and acquisition differ together.','Changes in gain under paired context transfer')

sim=read(R,'simulation_formal_replicates.csv'); assert len(sim)==2000 and (sim.simple_ci_accept==sim.context_005_accept).all()
scenes=[('null','standard','Uninformative'),('redundant','standard','Redundant'),('increment','standard','Informative'),('mispair','standard','Partial mismatch'),('domain_shift','visible','Visible shift'),('domain_shift','hidden','Hidden shift')]
sr=[]; vals=[]
for s,t,label in scenes:
    z=sim[(sim.scenario==s)&(sim.subtype==t)]; n=len(z); k=int(z.simple_ci_accept.sum()); p=k/n; den=1+1.96**2/n; center=(p+1.96**2/(2*n))/den; half=1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/den
    sr.append([label,n,k,p,center-half,center+half,z.target_gain.mean(),z.target_gain.std(ddof=1)/np.sqrt(n)]); vals.append(z.target_gain.to_numpy())
ss=pd.DataFrame(sr,columns=['condition','n','accepted','fraction','low','high','mean','mcse']); fig,ax=two(4.2,left=.19,wspace=1.25)
y=np.arange(6); ax[0].barh(y,ss.fraction,color=[GRAY,GRAY,TEAL,BLUE,GRAY,ORANGE],height=.55); ax[0].errorbar(ss.fraction,y,xerr=[ss.fraction-ss.low,ss.high-ss.fraction],fmt='none',c='#333333',capsize=2,lw=.8)
for r,i in zip(ss.itertuples(),y): ax[0].text(r.high+.04,i,f'{r.accepted}/{r.n}',va='center',fontsize=7)
ax[0].set_yticks(y,ss.condition); ax[0].invert_yaxis(); ax[0].set_xlim(0,1.35); ax[0].set_xticks([0,.5,1]); ax[0].set_xlabel('Fraction selecting fusion'); title(ax[0],'a','Selection decisions')
v=ax[1].violinplot(vals,positions=y,vert=False,widths=.7,showextrema=False)
for body in v['bodies']: body.set_facecolor('#CBD5DC'); body.set_edgecolor(GRAY); body.set_alpha(.8)
ax[1].errorbar(ss['mean'],y,xerr=1.96*ss.mcse,fmt='o',ms=3.5,color=ORANGE,lw=1,capsize=2); ax[1].axvline(0,c=GRAY,lw=.7,ls='--'); ax[1].set_yticks(y,ss.condition); ax[1].invert_yaxis(); ax[1].set_xlabel('Target Brier gain\nif fusion is used'); title(ax[1],'b','Target-gain distributions')
source('Figure_4','a',ss,'simulation_formal_replicates.csv','Independent simulation replicate, pooled settings'); source('Figure_4','b',sim,'simulation_formal_replicates.csv','Independent simulation replicate')
save(fig,'Figure_4','a, Fusion selection by either rule. Decisions are identical for every replicate. Bars give accepted fractions; labels give accepted/total counts and error bars Wilson 95% intervals. b, Distributions of target Brier gains when fusion is used in every replicate, regardless of the selection decision. Violin widths show kernel density; points and bars show mean ±1.96 Monte Carlo standard errors, not a prediction interval. Positive favors fusion. All 2,000 independent replicates are included: 400 each for uninformative, redundant, informative and partial-mismatch morphology; 200 each for visible and hidden shifts. Results pool prespecified sample-size and noise settings. Partial mismatch can retain signal. Realized selection-rule gains are separately reported in Tables S8–S9.','Selection decisions and target performance')

# Diagnostic uncertainty is retained where it is the scientific object.
rep=read(R,'repeat_aggregation_uncertainty.csv').query("family=='lr'"); res=read(R,'residual_sensitivity_summary.csv')
fig,ax=two(3.5,left=.18,wspace=.83)
intervals(ax[0],[a+'\n'+('Baseline' if m=='B' else 'Fusion') for a,m in zip(rep.assay,rep.model)],rep.point,rep.low,rep.high,ORANGE)
intervals(ax[1],res.site.str.replace('_',' '),res.mean_difference,res.ci95_low,res.ci95_high,TEAL)
ax[0].set_xlabel('Single-well − aggregate Brier loss'); ax[1].set_xlabel('Residual fusion − baseline Brier loss'); title(ax[0],'a','Repeat aggregation'); title(ax[1],'b','Residual-morphology sensitivity')
source('Figure_S1','a',rep,'repeat_aggregation_uncertainty.csv','Fixed-model paired prediction'); source('Figure_S1','b',res,'residual_sensitivity_summary.csv','Post hoc residual-model sensitivity')
save(fig,'Figure_S1','a, Change in Brier loss when single-well features replace replicate aggregates in the fixed aggregate-trained logistic models. b, Residual-morphology fusion compared with the baseline; positive values favor the baseline. Points and bars show paired estimates and 95% scaffold-bootstrap intervals (5,000 resamples). SR-ARE and SR-MMP include 85 and 95 strict held-out compounds; their union has 117 compounds/114 scaffolds. Residual-model parameters were specified after primary scoring. These diagnostics do not compare separately optimized single-well pipelines or establish biological independence.','Measurement and residualization sensitivities')

jump=read(F,'jump_fold_results.csv'); ev=read(F,'evebio_fixed_comparator_summary.csv'); jp=jump.pivot(index=['assay','seed','fold'],columns='model',values='average_precision').reset_index(); jp['increment']=jp['S+C+M']-jp['S+C']
fig,ax=two(5.8,left=.16,wspace=.85); fig.subplots_adjust(top=.91,bottom=.19,right=.90)
for i,assay in enumerate(['SR-ARE','SR-MMP','SR-p53','NR-ER-LBD']):
    z=jp[jp.assay==assay].increment; ax[0].scatter(z,i+np.linspace(-.13,.13,len(z)),s=14,c=GRAY,alpha=.7); ax[0].plot(z.mean(),i,'D',c=ORANGE,ms=4); ax[0].plot(z.median(),i,'|',c=BLUE,ms=12)
ax[0].axvline(0,c=GRAY,lw=.7,ls='--'); ax[0].set_yticks(range(4),['SR-ARE','SR-MMP','SR-p53','NR-ER-LBD']); ax[0].invert_yaxis(); ax[0].set_xlabel('AP difference versus\nstructure + count proxy'); ax[0].margins(y=.1); title(ax[0],'a','JUMP fold differences')
ax[0].plot([],[],'D',c=ORANGE,ms=4,label='Mean'); ax[0].plot([],[],'|',c=BLUE,ms=10,label='Median'); ax[0].set_ylim(4.15,-.45); ax[0].legend(loc='lower left',ncol=2,borderaxespad=.25)
ev=ev.sort_values('SCM_minus_SC_logit_median',ascending=False); mat=ev[['CM_minus_C_logit_median','SCM_minus_SC_logit_median']].to_numpy(); lim=np.max(np.abs(mat))
im=ax[1].imshow(mat,cmap='PuOr_r',vmin=-lim,vmax=lim,aspect='auto'); ax[1].set_yticks(range(len(ev)),ev.assay.str.replace('_Antagonist',' ant.').str.replace('_Agonist',' ago.'),fontsize=7); ax[1].set_xticks([0,1],['Count','Structure\n+ count']); ax[1].set_xlabel('Comparator'); title(ax[1],'b','EveBio baseline dependence')
for i in range(len(mat)):
    for j in range(2): ax[1].text(j,i,f'{mat[i,j]:+.2f}',ha='center',va='center',fontsize=6.5,color='white' if abs(mat[i,j])>.65*lim else '#222222')
cb=fig.colorbar(im,ax=ax[1],fraction=.055,pad=.07); cb.set_label('Median AP difference',fontsize=7); cb.ax.tick_params(labelsize=7)
source('Figure_S2','a',jp,'jump_fold_results.csv','Correlated seed–fold contrast'); source('Figure_S2','b',ev,'evebio_fixed_comparator_summary.csv','Endpoint median of nine correlated seed–fold contrasts')
save(fig,'Figure_S2','a, All 15 paired seed–fold AP differences per JUMP endpoint for morphology fusion versus a fixed structure-plus-count-proxy baseline. Gray points are fold differences, orange diamonds means and blue ticks medians. b, All 24 EveBio endpoint medians, each across three seeds and three scaffold folds. Columns compare count plus morphology versus count, and structure plus count plus morphology versus structure plus count. Colors are centered at zero; positive favors morphology. Values are differences on the original AP scale from logistic regression, not logit-transformed AP. Ant. and ago. denote antagonist and agonist tasks. Folds share training observations; neither their spread nor these medians are confidence intervals.','Dependence on the available comparator')

ch=read(F,'cellhealth_endpoint_context_summary.csv'); ab=read(F,'cellhealth_channel_ablation_summary.csv'); piv=ch.pivot(index='target',columns='split_type',values='morphology_minus_count').reset_index(); abb=ab.query("split_type=='cell_line_holdout'").copy()
fig,ax=two(4.7,left=.11,wspace=1.2); fig.subplots_adjust(right=.90); ax[0].scatter(piv.gene_holdout,piv.cell_line_holdout,c=BLUE,s=15,alpha=.7); ax[0].axhline(0,c=GRAY,lw=.7,ls='--'); ax[0].axvline(0,c=GRAY,lw=.7,ls='--'); ax[0].set_xlabel('Gene holdout:\nmorphology − count'); ax[0].set_ylabel('Cell-line holdout: morphology − count'); title(ax[0],'a','All cell-health endpoints')
channels=['DNA','ER','RNA','AGP','Mito']; mat=abb[['loss_when_removing_'+x for x in channels]].to_numpy(); lim=np.max(np.abs(mat)); im=ax[1].imshow(mat,cmap='PuOr_r',vmin=-lim,vmax=lim,aspect='auto')
label_map={'cc_all_n_spots_h2ax_mean':'All cells: H2AX spots','cc_cc_mitosis':'Mitosis','cc_early_mitosis_n_spots_h2ax_mean':'Early mitosis: H2AX spots','cc_g1_plus_g2_count':'G1 + G2 cell count','cc_polyploid_high_h2ax':'Polyploid: high H2AX','cc_polyploid_n_spots_h2ax_mean':'Polyploid: H2AX spots','cc_polyploid_n_spots_h2ax_per_nucleus_area_mean':'Polyploid: H2AX spots/area','vb_ros_mean':'ROS'}
assert set(abb.target)<=set(label_map),set(abb.target)-set(label_map)
labels=abb.target.map(label_map).tolist(); abb['display_label']=labels
ax[1].set_yticks(range(len(abb)),labels,fontsize=7); ax[1].set_xticks(range(5),channels,rotation=45,ha='right',rotation_mode='anchor'); ax[1].set_xlabel('Removed imaging channel'); title(ax[1],'b','Retained-endpoint ablation'); cb=fig.colorbar(im,ax=ax[1],fraction=.055,pad=.07); cb.set_label('Correlation loss',fontsize=7)
source('Figure_S3','a',piv,'cellhealth_endpoint_context_summary.csv','Cell-health endpoint'); source('Figure_S3','b',abb,'cellhealth_channel_ablation_summary.csv','Retained endpoint and omitted channel')
save(fig,'Figure_S3','a, Morphology-minus-count Spearman correlations for all 70 Cell Health endpoints under gene and cell-line holdout. Correlations are calculated within cell lines, then summarized. b, Correlation losses after removing each imaging channel during cell-line holdout, for all eight endpoints retained by exploratory criteria. Positive values indicate worse prediction after removal. The dataset represents 119 guides targeting 59 genes in three cell lines; model units are guide–line profiles, not cells. Endpoints and folds are correlated. No confidence interval or confirmatory test is assigned to the heatmap. Channel ablation locates predictive information, not a causal mechanism. DNA, nuclear channel; ER, endoplasmic reticulum; RNA, RNA-rich compartments; AGP, actin/Golgi/plasma membrane; Mito, mitochondria.','Perturbation information and channel contributions')

cr=read(C,'cardiac_slope_reliability_summary.csv').query("cohort=='main_119'"); cd=read(C,'cardiac_dose_leverage_summary.csv').query("cohort=='main_119'"); names=cr.pathway.drop_duplicates().tolist()
display={'Adrenergic signaling in cardiomyocytes':'Adrenergic signaling','Cardiac muscle contraction':'Cardiac contraction','Calcium signaling pathway':'Calcium signaling','cAMP signaling pathway':'cAMP signaling','cGMP-PKG signaling pathway':'cGMP–PKG signaling','Arrhythmogenic right ventricular cardiomyopathy':'Arrhythmogenic cardiomyopathy'}
fig,ax=two(4.7,left=.25,wspace=1.65); fig.subplots_adjust(bottom=.25,right=.90)
for i,an in enumerate(cr.analysis.unique()):
    z=cr[cr.analysis==an].set_index('pathway').loc[names]; ax[0].errorbar(z.median_pairwise_spearman,np.arange(9)+(i-.5)*.19,xerr=[z.median_pairwise_spearman-z.bootstrap_ci_low,z.bootstrap_ci_high-z.median_pairwise_spearman],fmt=['o','s'][i],ms=3,lw=.75,color=[GRAY,BLUE][i],label=an.replace('_',' ').capitalize())
ax[0].set_yticks(range(9),[display.get(n,n) for n in names],fontsize=7); ax[0].invert_yaxis(); ax[0].set_xlabel('Pairwise Spearman correlation'); ax[0].legend(loc='upper left',bbox_to_anchor=(0,-.18),fontsize=6.6); title(ax[0],'a','Slope repeatability')
dm=cd.pivot(index='pathway',columns='omission',values='spearman_with_full').loc[names,['without_low_0.2','without_mid_1','without_high_10']]; assert dm.min().min()>=0
im=ax[1].imshow(dm.to_numpy(),cmap='Blues',vmin=0,vmax=1,aspect='auto'); ax[1].set_yticks(range(9),[display.get(n,n) for n in names],fontsize=7); ax[1].set_xticks(range(3),['0.2','1','10']); ax[1].set_xlabel('Omitted dose (µM)'); title(ax[1],'b','Dose leverage')
for i in range(9):
    for j in range(3): ax[1].text(j,i,f'{dm.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if dm.iloc[i,j]>.55 else '#222222')
cb=fig.colorbar(im,ax=ax[1],fraction=.05,pad=.07); cb.set_label('Correlation with full slope',fontsize=7)
source('Figure_S4','a',cr,'cardiac_slope_reliability_summary.csv','Pathway summary in 119 compounds'); source('Figure_S4','b',cd,'cardiac_dose_leverage_summary.csv','Pathway and omitted dose in same 119 compounds')
save(fig,'Figure_S4','a, Median pairwise Spearman correlations among single-well slopes or leave-one-replicate-out aggregate slopes, with compound-bootstrap 95% intervals. b, Correlations between the full three-dose slope and slopes after omitting one concentration. Equally spaced columns denote categorical dose-omission conditions, not a continuous or logarithmic dose axis. All nine pathways retain the same 119 compounds; the full 462-compound cohort is reported in Tables S16–S18. Heatmap correlations are descriptive, without inferential intervals. Slopes summarize 48-hour, vehicle-centered transcriptomic responses against log10(1 + concentration in µM). The pathway targets share a strong common response component and are not independent biological confirmations.','Stability of the cardiomyocyte response target')

oc=read(C,'oasis_complete_case_pod.csv'); os=read(C,'oasis_paired_pod_summary.csv'); ratio=oc[['OASIS_ID','Compound_name']].copy(); fig,ax=two(3.7,left=.105,wspace=.65)
for key,color in zip(['MT','Cell_count','LDH'],[TEAL,BLUE,ORANGE]):
    ratio[key]=oc[key]/oc.Morphology; vals=np.sort(ratio[key]); assert np.all(vals>0); ax[0].step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',c=color,lw=1.2,label=key.replace('_',' '))
ax[0].set_xscale('log'); ax[0].set_xticks([.01,.1,1,10,100,1000],['0.01','0.1','1','10','100','1,000']); ax[0].axvline(1,c=GRAY,lw=.7,ls='--'); ax[0].set(xlabel='Response-concentration ratio\n(comparator / morphology)',ylabel='Cumulative fraction of compounds',ylim=(0,1.02)); ax[0].legend(loc='lower right'); title(ax[0],'a','All paired complete cases')
intervals(ax[1],os.comparator.str.replace('_',' '),os.geometric_mean_ratio_other_over_morphology,os.geometric_ratio_bootstrap_low,os.geometric_ratio_bootstrap_high,TEAL); ax[1].set_xscale('log'); ax[1].axvline(1,c=GRAY,lw=.7,ls='--'); ax[1].set_xticks([1,2,5,10],['1','2','5','10']); ax[1].set_xlabel('Geometric mean concentration ratio\n(comparator / morphology)'); title(ax[1],'b','Paired concentration contrast')
source('Figure_S5','a',ratio,'oasis_complete_case_pod.csv','Compound complete in all four modalities'); source('Figure_S5','b',os,'oasis_paired_pod_summary.csv','Paired compound-bootstrap estimate')
save(fig,'Figure_S5','a, Empirical cumulative distributions of comparator/morphology response-concentration ratios for all 121 compounds with detected measurements in all four modalities. Each step represents observed compounds; logarithmic horizontal axes retain the full ratio range. b, Geometric mean paired ratios with compound-bootstrap 95% intervals. A ratio above one indicates morphological change at a lower concentration, not earlier time. MT is metabolic activity measured by RealTime-Glo; LDH is lactate dehydrogenase release. Complete cases exclude compounds lacking a detected response, so these estimates do not describe all 1,085 screened compounds or establish target specificity or clinical toxicity.','Response-concentration distributions in OASIS')

(W/'figure_legends.json').write_text(json.dumps(legends,ensure_ascii=False,indent=2),'utf-8')
(W/'figure_panel_registry.json').write_text(json.dumps(registry,ensure_ascii=False,indent=2),'utf-8')
(W/'figure_input_hashes.json').write_text(json.dumps(inputs,indent=2),'utf-8')
manifest=P/'preprocessing/EUOS_profiles_intake_2026-09-21_v1/profile_file_manifest.csv'; shutil.copy2(manifest,SD/manifest.name)
with zipfile.ZipFile(W/'Figure_Source_Data_v8.1.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(SD.glob('*.csv')): archive.write(p,p.name)
print(json.dumps({'figures':len(legends),'panels':len(registry),'delivery':str(D)},ensure_ascii=False))
