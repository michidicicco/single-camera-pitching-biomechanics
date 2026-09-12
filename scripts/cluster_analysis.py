#!/usr/bin/env python
"""Robustness analyses for the fixed model-ready biomechanical matrix."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
META={'video_file','pitcher_id','pitch_id','unique_pitch_id','measurement_qc_status','model_eligible','relative_mechanical_band_pooled','relative_mechanical_band_within_pitcher','relative_mechanical_score_pooled','relative_mechanical_score_within_pitcher','mechanical_flag_count','mechanical_flag_profile'}
def args():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output-dir','--output_dir',dest='output_dir',required=True);p.add_argument('--full-data');p.add_argument('--max-k',type=int,default=6);p.add_argument('--seed',type=int,default=13);p.add_argument('--seed-stability-runs',type=int,default=30);p.add_argument('--balanced-resamples',type=int,default=100);p.add_argument('--balanced-pitches-per-pitcher',type=int,default=0);return p.parse_args()
def standardize(X):
 m=X.mean(0);s=X.std(0);s=np.where((s==0)|~np.isfinite(s),1.,s);return (X-m)/s
def d2(X,C):return((X[:,None,:]-C[None,:,:])**2).sum(2)
def km(X,k,seed,n_init=50,max_iter=300):
 master=np.random.default_rng(seed);best=None;bi=np.inf;n=len(X)
 for _ in range(n_init):
  r=np.random.default_rng(int(master.integers(0,2**31-1)));C=X[r.choice(n,k,replace=False)].copy();lab=np.full(n,-1)
  for it in range(1,max_iter+1):
   nl=d2(X,C).argmin(1);NC=C.copy()
   for j in range(k):NC[j]=X[nl==j].mean(0) if np.any(nl==j) else X[r.integers(0,n)]
   if np.array_equal(nl,lab) and np.allclose(NC,C):lab,C=nl,NC;break
   lab,C=nl,NC
  iner=float(np.min(d2(X,C),axis=1).sum())
  if iner<bi:best=(lab.copy(),C.copy(),iner,it);bi=iner
 return best
def sil(X,lab):
 D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(2));u=np.unique(lab);v=[]
 for i in range(len(X)):
  same=lab==lab[i];same[i]=False;a=float(D[i,same].mean()) if same.sum() else 0.;b=min(float(D[i,lab==x].mean()) for x in u if x!=lab[i]);den=max(a,b);v.append((b-a)/den if den else 0.)
 return float(np.mean(v))
def ari(a,b):
 from sklearn.metrics import adjusted_rand_score
 return float(adjusted_rand_score(a,b))
def select(Z,maxk,seed):
 rows=[]
 for k in range(2,min(maxk,len(Z)-1)+1):
  l,c,i,it=km(Z,k,seed,50,300);rows.append({'k':k,'silhouette_score':sil(Z,l),'inertia':i,'n_iter':it})
 return pd.DataFrame(rows)
def seed_stability(Z,k,seed,nruns):
 labs=[];rows=[]
 for off in range(nruns):
  s=seed+off;l,c,i,it=km(Z,k,s,20,300);labs.append(l);rows.append({'seed':s,'silhouette_score':sil(Z,l),'inertia':i,'n_iter':it})
 vals=[ari(labs[i],labs[j]) for i in range(len(labs)) for j in range(i+1,len(labs))];summ={'mean_pairwise_ARI':float(np.mean(vals)),'min_pairwise_ARI':float(np.min(vals)),'max_pairwise_ARI':float(np.max(vals))};df=pd.DataFrame(rows)
 for k2,v in summ.items():df[k2]=v
 return df,summ
def groups(features):
 g={'all_features':features,'throwing_arm':[f for f in features if any(x in f.lower() for x in ['elbow','wrist','upper_arm','forearm'])],'trunk_hip_shoulder':[f for f in features if any(x in f.lower() for x in ['trunk','hip_shoulder','shoulder'])],'lower_body_stride':[f for f in features if any(x in f.lower() for x in ['stride','lead_knee','trail_knee','ankle'])],'relative_timing':[f for f in features if 'to_release_ms' in f.lower() or f.lower()=='ffc_to_release_ms']};return{k:v for k,v in g.items() if len(v)>=2}
def ablation(X,out,maxk,seed):
 rows=[]
 for name,cols in groups(list(X.columns)).items():
  Z=standardize(X[cols].to_numpy(float));sel=select(Z,maxk,seed);best=sel.loc[sel.silhouette_score.idxmax()];k=int(best.k);st,ss=seed_stability(Z,k,seed,20);S=np.linalg.svd(Z,full_matrices=False,compute_uv=False);ev=(S**2)/max(len(Z)-1,1);ex=ev/ev.sum();rows.append({'feature_group':name,'n_features':len(cols),'best_k':k,'best_silhouette':float(best.silhouette_score),'PC1_variance':float(ex[0]),'PC2_variance':float(ex[1]),'PC1_PC2_combined_variance':float(ex[:2].sum()),'mean_seed_stability_ARI':ss['mean_pairwise_ARI'],'min_seed_stability_ARI':ss['min_pairwise_ARI'],'features_used':';'.join(cols)});sel.to_csv(out/f'ablation_{name}_kmeans_selection.csv',index=False);st.to_csv(out/f'ablation_{name}_seed_stability.csv',index=False)
 return pd.DataFrame(rows)
def main():
 a=args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);df=pd.read_csv(a.input);features=[c for c in df.columns if c not in META and pd.to_numeric(df[c],errors='coerce').notna().any()];X=df[features].apply(pd.to_numeric,errors='coerce');
 if X.isna().any().any():raise ValueError('Model-ready data contain missing values.')
 Z=standardize(X.to_numpy(float));sel=select(Z,a.max_k,a.seed);best=sel.loc[sel.silhouette_score.idxmax()];bk=int(best.k);base,_,_,_=km(Z,bk,a.seed,100,500);bs=sil(Z,base);st,ss=seed_stability(Z,bk,a.seed,a.seed_stability_runs)
 ids=df[[c for c in ['unique_pitch_id','video_file','pitcher_id','pitch_id'] if c in df]].copy();ids['cluster']=base;ids.to_csv(out/'cluster_assignments.csv',index=False);sel.to_csv(out/'kmeans_model_selection.csv',index=False);st.to_csv(out/'initialization_stability.csv',index=False)
 abl=ablation(X,out,a.max_k,a.seed);abl.to_csv(out/'feature_group_analysis.csv',index=False)
 lo=[];bal=[]
 if 'pitcher_id' in df:
  p=df.pitcher_id.astype(str)
  for pid in sorted(p.unique()):
   keep=p!=pid;Zs=standardize(X.loc[keep].to_numpy(float));lab,_,_,_=km(Zs,bk,a.seed,50,300);lo.append({'left_out_pitcher':pid,'n_remaining':int(keep.sum()),'ARI_vs_full_solution_on_remaining_rows':ari(base[keep.to_numpy()],lab),'silhouette':sil(Zs,lab),'note':'fixed_k_equal_to_primary_solution'})
  counts=p.value_counts();ne=a.balanced_pitches_per_pitcher or int(counts.min());rng=np.random.default_rng(a.seed)
  for rep in range(a.balanced_resamples):
   idx=[]
   for pid in sorted(counts.index):
    choices=np.where((p==pid).to_numpy())[0];idx.extend(rng.choice(choices,size=min(ne,len(choices)),replace=False).tolist())
   idx=np.array(sorted(idx));Zs=standardize(X.iloc[idx].to_numpy(float));lab,_,_,_=km(Zs,bk,a.seed+rep,30,300);bal.append({'resample':rep,'n_per_pitcher':ne,'n_rows':len(idx),'ARI_vs_full_solution_on_sampled_rows':ari(base[idx],lab),'silhouette':sil(Zs,lab)})
 pd.DataFrame(lo).to_csv(out/'leave_one_pitcher_out.csv',index=False);pd.DataFrame(bal).to_csv(out/'pitcher_balanced_resampling.csv',index=False);pd.crosstab(df.pitcher_id,base).to_csv(out/'pitcher_by_cluster.csv')
 lodf=pd.DataFrame(lo);bdf=pd.DataFrame(bal);summary={'input_model_ready':str(a.input),'rows':len(df),'features':len(features),'best_k':bk,'silhouette':bs,'seed_stability_mean_pairwise_ARI':ss['mean_pairwise_ARI'],'seed_stability_min_pairwise_ARI':ss['min_pairwise_ARI'],'leave_one_pitcher_out_mean_ARI_vs_full_remaining':float(lodf.ARI_vs_full_solution_on_remaining_rows.mean()),'pitcher_balanced_resampling_mean_ARI_vs_full_sampled':float(bdf.ARI_vs_full_solution_on_sampled_rows.mean()),'pitcher_balanced_resampling_min_ARI_vs_full_sampled':float(bdf.ARI_vs_full_solution_on_sampled_rows.min())};(out/'analysis_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
