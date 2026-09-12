#!/usr/bin/env python
"""Prepare the primary biomechanical matrix and run descriptive PCA/K-means.

The public implementation uses the 36 prespecified biomechanical candidates from
the associated study. Identifier, acquisition, QC, event-detector, and derived
relative-mechanics fields are never model inputs. Redundancy is reduced at
|r| >= 0.95 using the documented deterministic rule. PCA is descriptive; K-means
is run on standardized retained original features.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CANDIDATE_FEATURES = [
'ffc_to_release_ms','stride_length_norm_at_ffc','stride_length_norm_at_release','stride_length_norm_max_ffc_to_release',
'elbow_angle_deg_at_ffc','elbow_angle_deg_at_release','elbow_angle_deg_mean_ffc_to_release','elbow_angle_deg_range_ffc_to_release',
'elbow_ang_vel_max_ffc_to_release','elbow_ang_vel_mean_ffc_to_release',
'wrist_speed_norm_bh_per_s_max_ffc_to_release','wrist_speed_norm_bh_per_s_mean_ffc_to_release','wrist_speed_norm_bh_per_s_at_release',
'wrist_speed_max_ffc_to_release','wrist_speed_mean_ffc_to_release','wrist_speed_at_release',
'upper_arm_angle_deg_at_release','forearm_angle_deg_at_release',
'trunk_tilt_deg_at_ffc','trunk_tilt_deg_at_release','trunk_tilt_deg_mean_ffc_to_release','trunk_tilt_deg_range_ffc_to_release','trunk_tilt_vel_max_ffc_to_release',
'hip_shoulder_sep_deg_at_ffc','hip_shoulder_sep_deg_at_release','hip_shoulder_sep_deg_max_ffc_to_release','hip_shoulder_sep_deg_mean_ffc_to_release','hip_shoulder_sep_vel_max_ffc_to_release',
'lead_knee_angle_deg_at_ffc','lead_knee_angle_deg_at_release','lead_knee_extension_deg_ffc_to_release',
'trail_knee_angle_deg_at_ffc','trail_knee_angle_deg_at_release',
'max_sep_to_release_ms','max_elbow_flex_to_release_ms','max_trunk_tilt_vel_to_release_ms']
META = ['video_file','pitcher_id','pitch_id','unique_pitch_id','measurement_qc_status','model_eligible','relative_mechanical_band_pooled','relative_mechanical_band_within_pitcher','relative_mechanical_score_pooled','relative_mechanical_score_within_pitcher','mechanical_flag_count','mechanical_flag_profile']

def args():
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output-dir','--output_dir',dest='output_dir',required=True); p.add_argument('--corr-threshold',type=float,default=.95); p.add_argument('--max-feature-missing-frac',type=float,default=.10); p.add_argument('--max-k',type=int,default=6); p.add_argument('--random-state',type=int,default=13); return p.parse_args()
def uid(df):
 if 'unique_pitch_id' in df: return df.unique_pitch_id.astype(str)
 if {'pitcher_id','pitch_id'}<=set(df.columns): return df.pitcher_id.astype(str)+'_'+df.pitch_id.astype(str)
 return df.video_file.astype(str).str.replace(r'\.[^.]+$','',regex=True)
def eligible(df):
 if 'model_eligible' in df: return pd.to_numeric(df.model_eligible,errors='coerce').fillna(0).astype(int).eq(1)
 if 'measurement_qc_status' in df: return ~df.measurement_qc_status.astype(str).str.lower().eq('exclude')
 return pd.Series(True,index=df.index)
def corr_cleanup(X,threshold,missing):
 work=X.copy(); decisions=[]
 while work.shape[1]>1:
  corr=work.corr(min_periods=3).abs(); corr=corr.mask(np.eye(corr.shape[0],dtype=bool)); mx=np.nanmax(corr.to_numpy())
  if not np.isfinite(mx) or mx<threshold: break
  pairs=[]
  for i,a in enumerate(corr.columns):
   for j in range(i+1,len(corr.columns)):
    b=corr.columns[j]; v=corr.loc[a,b]
    if np.isfinite(v) and np.isclose(v,mx): pairs.append((a,b,float(v)))
  a,b,v=sorted(pairs)[0]
  def key(c):
   mc=float(corr[c].drop(labels=[c],errors='ignore').mean(skipna=True)); mc=mc if np.isfinite(mc) else 0.; return (float(missing.get(c,1.)),mc,c)
  keep,remove=(a,b) if key(a)<=key(b) else (b,a)
  decisions.append({'removed_feature':remove,'kept_feature':keep,'abs_corr':v,'removal_reason':'high_correlation; lower missing fraction, then lower mean absolute correlation, then alphabetical tie-break'})
  work=work.drop(columns=[remove])
 return work,pd.DataFrame(decisions)
def d2(X,C): return ((X[:,None,:]-C[None,:,:])**2).sum(axis=2)
def kmeans(X,k,seed,n_init=50,max_iter=300):
 master=np.random.default_rng(seed); n=len(X); best=None; bi=np.inf
 for _ in range(n_init):
  rng=np.random.default_rng(int(master.integers(0,2**31-1))); C=X[rng.choice(n,k,replace=False)].copy(); lab=np.full(n,-1)
  for it in range(1,max_iter+1):
   nl=d2(X,C).argmin(1); NC=C.copy()
   for j in range(k): NC[j]=X[nl==j].mean(0) if np.any(nl==j) else X[rng.integers(0,n)]
   if np.array_equal(nl,lab) and np.allclose(NC,C): lab,C=nl,NC; break
   lab,C=nl,NC
  inertia=float(np.min(d2(X,C),axis=1).sum())
  if inertia<bi: best=(lab.copy(),C.copy(),inertia,it); bi=inertia
 return best
def silhouette(X,lab):
 D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(2)); vals=[]; u=np.unique(lab)
 for i in range(len(X)):
  same=lab==lab[i]; same[i]=False; a=float(D[i,same].mean()) if same.sum() else 0.; b=min(float(D[i,lab==x].mean()) for x in u if x!=lab[i]); den=max(a,b); vals.append((b-a)/den if den else 0.)
 return float(np.mean(vals))
def main():
 a=args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(a.input); df['unique_pitch_id']=uid(df); em=eligible(df); edf=df.loc[em].copy()
 missing=[f for f in CANDIDATE_FEATURES if f not in df.columns]
 if missing: raise ValueError('Missing prespecified biomechanical candidates: '+', '.join(missing))
 disp=[]; pre=[]; miss={}
 for f in CANDIDATE_FEATURES:
  s=pd.to_numeric(edf[f],errors='coerce'); mf=float(s.isna().mean()); miss[f]=mf
  if mf>a.max_feature_missing_frac: disp.append({'feature':f,'status':'excluded','reason':'missingness','missing_fraction':mf})
  elif s.dropna().nunique()<=1: disp.append({'feature':f,'status':'excluded','reason':'zero_variance','missing_fraction':mf})
  else: pre.append(f); disp.append({'feature':f,'status':'candidate','reason':'eligible_biomechanical_candidate','missing_fraction':mf})
 Xpre=edf[pre].apply(pd.to_numeric,errors='coerce'); Xcorr,dec=corr_cleanup(Xpre,a.corr_threshold,pd.Series(miss)); removed=set(dec.removed_feature) if len(dec) else set()
 for r in disp:
  if r['feature'] in removed: r['status']='excluded'; r['reason']='high_correlation'
  elif r['feature'] in Xcorr.columns: r['status']='retained'; r['reason']='primary_model_feature'
 complete=Xcorr.notna().all(1); X=Xcorr.loc[complete].copy(); rows=edf.loc[complete].copy()
 pd.DataFrame(disp).to_csv(out/'feature_disposition.csv',index=False); dec.to_csv(out/'high_correlation_removal_log.csv',index=False); pd.DataFrame({'feature':X.columns}).to_csv(out/'retained_features.csv',index=False)
 excluded=edf.loc[~complete,[c for c in ['unique_pitch_id','video_file','pitcher_id','pitch_id'] if c in edf]].copy(); excluded['reason']='missing_one_or_more_retained_model_features'; excluded.to_csv(out/'model_rows_excluded_complete_case.csv',index=False)
 pd.concat([rows[[c for c in META if c in rows]].reset_index(drop=True),X.reset_index(drop=True)],axis=1).to_csv(out/'model_ready.csv',index=False)
 Z=StandardScaler().fit_transform(X); nc=min(5,*Z.shape); pca=PCA(n_components=nc); pcs=pca.fit_transform(Z)
 pd.DataFrame({'PC':[f'PC{i+1}' for i in range(nc)],'explained_variance_ratio':pca.explained_variance_ratio_,'cumulative_explained_variance_ratio':np.cumsum(pca.explained_variance_ratio_)}).to_csv(out/'pca_explained_variance.csv',index=False)
 pd.DataFrame(pca.components_.T,index=X.columns,columns=[f'PC{i+1}' for i in range(nc)]).to_csv(out/'pca_loadings.csv')
 score=rows[[c for c in ['unique_pitch_id','video_file','pitcher_id','pitch_id'] if c in rows]].reset_index(drop=True)
 for i in range(nc): score[f'PC{i+1}']=pcs[:,i]
 score.to_csv(out/'pca_scores.csv',index=False)
 sels=[]; maxk=min(a.max_k,len(X)-1)
 for k in range(2,maxk+1):
  lab,C,iner,it=kmeans(Z,k,a.random_state,50,300); sels.append({'k':k,'silhouette_score':silhouette(Z,lab),'inertia':iner,'n_iter':it})
 sel=pd.DataFrame(sels); sel.to_csv(out/'kmeans_model_selection.csv',index=False); best=sel.loc[sel.silhouette_score.idxmax()]; bk=int(best.k); lab,C,iner,it=kmeans(Z,bk,a.random_state,100,500)
 assign=score.copy(); assign['cluster']=lab; assign.to_csv(out/'cluster_assignments.csv',index=False)
 xo=X.copy(); xo['cluster']=lab; xo.groupby('cluster').mean(numeric_only=True).to_csv(out/'cluster_feature_means_original_units.csv'); z=pd.DataFrame(Z,columns=X.columns); z['cluster']=lab; z.groupby('cluster').mean(numeric_only=True).to_csv(out/'cluster_feature_means_zscore.csv')
 if pcs.shape[1]>=2:
  plt.figure(figsize=(10,7)); [plt.scatter(pcs[lab==c,0],pcs[lab==c,1],label=f'Cluster {c}') for c in sorted(np.unique(lab))]; plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)'); plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'); plt.legend(); plt.tight_layout(); plt.savefig(out/'pca_clusters.png',dpi=300); plt.close()
 summary={'input_file':str(a.input),'input_rows':int(len(df)),'qc_eligible_rows':int(em.sum()),'complete_case_model_rows':int(len(X)),'candidate_features_initial':len(CANDIDATE_FEATURES),'retained_model_features':int(X.shape[1]),'max_feature_missing_frac':a.max_feature_missing_frac,'correlation_threshold':a.corr_threshold,'scaling':'sklearn StandardScaler','pca':'sklearn PCA; descriptive only','kmeans':'custom NumPy K-means on standardized retained original features','kmeans_model_selection_n_init':50,'kmeans_refit_n_init':100,'random_state':a.random_state,'candidate_k_min':2,'candidate_k_max':maxk,'best_k':bk,'best_silhouette':float(best.silhouette_score),'refit_inertia':float(iner),'refit_iterations':int(it)}
 (out/'analysis_summary.json').write_text(json.dumps(summary,indent=2)); (out/'analysis_summary.txt').write_text('\n'.join(f'{k}: {v}' for k,v in summary.items())+'\n')
 print('Primary analysis complete.'); print(f'Retained features: {len(X.columns)}'); print(f'Best k: {bk}'); print(f'Silhouette: {float(best.silhouette_score):.10f}')
if __name__=='__main__': main()
