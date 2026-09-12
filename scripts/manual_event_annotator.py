#!/usr/bin/env python
from __future__ import annotations
import argparse
import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_FEATURES = PROJECT_ROOT / 'outputs' / 'features' / 'pitch_features_master.csv'
DEFAULT_VIDEO_DIR = PROJECT_ROOT / 'videos'
DEFAULT_OUT = PROJECT_ROOT / 'metadata' / 'manual_event_annotations.csv'
WINDOW_NAME = 'Manual Event Annotation - BLINDED'

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Interactively annotate FFC and ball release directly from pitching video.')
    p.add_argument('--features', type=Path, default=DEFAULT_FEATURES)
    p.add_argument('--video-dir', type=Path, default=DEFAULT_VIDEO_DIR)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--n-per-pitcher', type=int, default=5, help='Number of pitches sampled per pitcher. Use 0 to annotate every available pitch.')
    p.add_argument('--seed', type=int, default=13, help='Random seed used for reproducible balanced pitch selection.')
    p.add_argument('--overwrite-selection', action='store_true', help='Ignore an existing annotation file when selecting the validation subset.')
    return p.parse_args()

def make_unique_pitch_id(df: pd.DataFrame) -> pd.Series:
    if 'unique_pitch_id' in df.columns:
        return df['unique_pitch_id'].astype(str)
    if {'pitcher_id', 'pitch_id'}.issubset(df.columns):
        return df['pitcher_id'].astype(str) + '_' + df['pitch_id'].astype(str)
    if 'video_file' in df.columns:
        return df['video_file'].astype(str).str.replace('\.[^.]+$', '', regex=True)
    return pd.Series([f'row_{i:04d}' for i in range(len(df))], index=df.index)

def load_candidates(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f'Feature table not found:\n{features_path}\n\nRun the feature pipeline first.')
    df = pd.read_csv(features_path)
    if 'video_file' not in df.columns:
        raise ValueError('Feature table must contain video_file.')
    df = df.copy(); df['unique_pitch_id'] = make_unique_pitch_id(df)
    if 'pitcher_id' not in df.columns: df['pitcher_id'] = 'unknown_pitcher'
    if 'pitch_id' not in df.columns: df['pitch_id'] = df['unique_pitch_id']
    return df[['unique_pitch_id','pitcher_id','pitch_id','video_file']].drop_duplicates('unique_pitch_id').reset_index(drop=True)

def balanced_sample(df: pd.DataFrame, n_per_pitcher: int, seed: int) -> pd.DataFrame:
    if n_per_pitcher <= 0:
        return df.sort_values(['pitcher_id','unique_pitch_id']).reset_index(drop=True)
    selected=[]
    for _,group in df.groupby('pitcher_id',sort=True):
        selected.append(group.sample(n=min(n_per_pitcher,len(group)),random_state=seed))
    return pd.concat(selected,ignore_index=True).sort_values(['pitcher_id','unique_pitch_id']).reset_index(drop=True) if selected else df.iloc[0:0].copy()

def load_existing(path: Path) -> pd.DataFrame:
    columns=['unique_pitch_id','video_file','pitcher_id','pitch_id','manual_ffc_frame_idx','manual_release_frame_idx','annotator','notes']
    if not path.exists(): return pd.DataFrame(columns=columns)
    df=pd.read_csv(path)
    for col in columns:
        if col not in df.columns: df[col]=np.nan
    return df[columns].copy()

def save_annotation(out_path: Path, existing: pd.DataFrame, record: dict) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True,exist_ok=True); df=existing.copy(); uid=str(record['unique_pitch_id'])
    if 'unique_pitch_id' in df.columns and (df['unique_pitch_id'].astype(str)==uid).any():
        mask=df['unique_pitch_id'].astype(str)==uid
        for key,value in record.items(): df.loc[mask,key]=value
    else: df=pd.concat([df,pd.DataFrame([record])],ignore_index=True)
    df=df.sort_values(['pitcher_id','unique_pitch_id'],na_position='last'); df.to_csv(out_path,index=False); return df

def read_frame(cap: cv2.VideoCapture, frame_idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame_idx)); ok,frame=cap.read(); return frame if ok else None

def resize_for_display(frame,max_width=1400,max_height=850):
    h,w=frame.shape[:2]; scale=min(max_width/w,max_height/h,1.0)
    return cv2.resize(frame,(int(round(w*scale)),int(round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1.0 else frame

def put_text(img,text,y,scale=0.65,thickness=2):
    cv2.putText(img,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),thickness,cv2.LINE_AA)
    cv2.putText(img,text,(18,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),max(1,thickness-1),cv2.LINE_AA)

def draw_overlay(frame,row,index,total,frame_idx,n_frames,fps,ffc_idx,release_idx,playing):
    img=resize_for_display(frame.copy()); h,w=img.shape[:2]; overlay=img.copy(); cv2.rectangle(overlay,(0,0),(w,178),(0,0,0),-1); img=cv2.addWeighted(overlay,0.55,img,0.45,0)
    put_text(img,f"Pitch {index+1}/{total}   |   {row['unique_pitch_id']}   |   {row['video_file']}",28,0.68)
    time_ms=frame_idx/fps*1000.0 if fps>0 else float('nan'); put_text(img,f'Frame: {frame_idx}/{max(n_frames-1,0)}   Time: {time_ms:.1f} ms   FPS: {fps:.3f}',57)
    put_text(img,f"Manual FFC [F]: {'NOT SET' if ffc_idx is None else ffc_idx}     Manual Release [R]: {'NOT SET' if release_idx is None else release_idx}",86)
    put_text(img,f"{'PLAYING' if playing else 'PAUSED'} | A/D = +/-1 frame | J/L = +/-5 | SPACE = play/pause | F = FFC | R = release",116,0.56)
    put_text(img,'ENTER = save/next | BACKSPACE = clear event at current frame | Q/ESC = save progress & quit',143,0.54)
    put_text(img,'BLINDED VALIDATION: algorithm event estimates are intentionally hidden.',169,0.54)
    if n_frames>1:
        bar_y=h-25; cv2.rectangle(img,(20,bar_y),(w-20,bar_y+8),(80,80,80),-1); x=int(20+frame_idx/(n_frames-1)*(w-40)); cv2.rectangle(img,(20,bar_y),(x,bar_y+8),(220,220,220),-1)
        if ffc_idx is not None:
            fx=int(20+ffc_idx/(n_frames-1)*(w-40)); cv2.line(img,(fx,bar_y-8),(fx,bar_y+16),(0,255,255),2)
        if release_idx is not None:
            rx=int(20+release_idx/(n_frames-1)*(w-40)); cv2.line(img,(rx,bar_y-8),(rx,bar_y+16),(255,255,0),2)
    return img

def annotate_video(video_path: Path,row: pd.Series,index: int,total: int,prior=None):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f'Could not open video: {video_path}')
    fps=float(cap.get(cv2.CAP_PROP_FPS)); n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps<=0: cap.release(); raise RuntimeError(f'Invalid FPS metadata for {video_path.name}. Do not manually validate timing until the actual FPS is known.')
    if n_frames<=0: cap.release(); raise RuntimeError(f'Invalid frame count for {video_path.name}.')
    frame_idx=0; ffc_idx=None; release_idx=None
    if prior is not None:
        fv=prior.get('manual_ffc_frame_idx',np.nan); rv=prior.get('manual_release_frame_idx',np.nan)
        if pd.notna(fv): ffc_idx=int(fv); frame_idx=ffc_idx
        if pd.notna(rv): release_idx=int(rv)
    playing=False; last_tick=cv2.getTickCount(); cv2.namedWindow(WINDOW_NAME,cv2.WINDOW_NORMAL)
    while True:
        frame=read_frame(cap,frame_idx)
        if frame is None:
            frame_idx=max(0,min(frame_idx,n_frames-1)); frame=read_frame(cap,frame_idx)
            if frame is None: cap.release(); raise RuntimeError(f'Could not read frame {frame_idx} from {video_path.name}.')
        cv2.imshow(WINDOW_NAME,draw_overlay(frame,row,index,total,frame_idx,n_frames,fps,ffc_idx,release_idx,playing)); key=cv2.waitKeyEx(max(1,int(round(1000/fps))) if playing else 30)
        if playing:
            now=cv2.getTickCount(); elapsed=(now-last_tick)/cv2.getTickFrequency()
            if elapsed>=1.0/fps:
                frame_idx+=1
                if frame_idx>=n_frames: frame_idx=n_frames-1; playing=False
                last_tick=now
        if key==-1: continue
        k=key&255
        if k in (ord('q'),ord('Q'),27): cap.release(); return {'action':'quit','ffc':ffc_idx,'release':release_idx}
        if k==32: playing=not playing; last_tick=cv2.getTickCount(); continue
        if k in (ord('a'),ord('A')): playing=False; frame_idx=max(0,frame_idx-1); continue
        if k in (ord('d'),ord('D')): playing=False; frame_idx=min(n_frames-1,frame_idx+1); continue
        if k in (ord('j'),ord('J')): playing=False; frame_idx=max(0,frame_idx-5); continue
        if k in (ord('l'),ord('L')): playing=False; frame_idx=min(n_frames-1,frame_idx+5); continue
        if k in (ord('f'),ord('F')): playing=False; ffc_idx=int(frame_idx); continue
        if k in (ord('r'),ord('R')): playing=False; release_idx=int(frame_idx); continue
        if k in (8,127):
            playing=False
            if ffc_idx==frame_idx: ffc_idx=None
            if release_idx==frame_idx: release_idx=None
            continue
        if k in (13,10):
            if ffc_idx is None or release_idx is None: print(f"Cannot save {row['unique_pitch_id']}: both FFC and release must be marked."); continue
            if release_idx<=ffc_idx: print(f"Cannot save {row['unique_pitch_id']}: release must occur after FFC."); continue
            cap.release(); return {'action':'save','ffc':int(ffc_idx),'release':int(release_idx)}

def main():
    args=parse_args(); candidates=load_candidates(args.features); existing=load_existing(args.out); sample=balanced_sample(candidates,args.n_per_pitcher,args.seed)
    print('='*72); print('BLINDED MANUAL EVENT ANNOTATOR'); print('='*72); print(f'Feature table : {args.features}'); print(f'Video folder  : {args.video_dir}'); print(f'Output CSV    : {args.out}'); print(f'Seed          : {args.seed}'); print(f'Pitches chosen: {len(sample)}'); print('\nAlgorithm estimates are NOT shown.'); print('='*72)
    completed=0
    for i,row in sample.iterrows():
        uid=str(row['unique_pitch_id']); prior=None
        if not existing.empty:
            match=existing[existing['unique_pitch_id'].astype(str)==uid]
            if not match.empty:
                prior=match.iloc[0]
                if pd.notna(prior.get('manual_ffc_frame_idx',np.nan)) and pd.notna(prior.get('manual_release_frame_idx',np.nan)):
                    print(f'Skipping already annotated: {uid}'); completed+=1; continue
        video_path=args.video_dir/str(row['video_file'])
        if not video_path.exists(): print(f'WARNING: video not found, skipping: {video_path}'); continue
        result=annotate_video(video_path,row,i,len(sample),prior)
        if result['ffc'] is not None or result['release'] is not None:
            existing=save_annotation(args.out,existing,{'unique_pitch_id':uid,'video_file':row['video_file'],'pitcher_id':row['pitcher_id'],'pitch_id':row['pitch_id'],'manual_ffc_frame_idx':result['ffc'],'manual_release_frame_idx':result['release'],'annotator':'','notes':''})
        if result['action']=='save': completed+=1; print(f"Saved {uid}: FFC={result['ffc']}, Release={result['release']}")
        if result['action']=='quit': print('\nProgress saved. Exiting.'); break
    cv2.destroyAllWindows(); print(f'Completed this validation subset: {completed}/{len(sample)}'); print(f'Saved to: {args.out}')
if __name__=='__main__': main()
