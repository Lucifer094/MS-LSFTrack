from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .io import boxes_to_centers


def filter_frames(arr: np.ndarray, start: Optional[int], end: Optional[int]) -> np.ndarray:
    if arr.size == 0:
        return arr
    mask = np.ones(len(arr), dtype=bool)
    if start is not None:
        mask &= arr[:,0] >= start
    if end is not None:
        mask &= arr[:,0] <= end
    return arr[mask]


def filter_roi(arr: np.ndarray, roi: Optional[Tuple[float,float,float,float]]) -> np.ndarray:
    if arr.size == 0 or roi is None:
        return arr
    x1,y1,x2,y2=roi
    c=boxes_to_centers(arr)
    m=(c[:,0]>=x1)&(c[:,0]<=x2)&(c[:,1]>=y1)&(c[:,1]<=y2)
    return arr[m]


def setup_ax(ax, img_w=256, img_h=256, title='', hide_axis=True, roi=None):
    if roi:
        x1,y1,x2,y2=roi; ax.set_xlim(x1,x2); ax.set_ylim(y2,y1)
    else:
        ax.set_xlim(0,img_w); ax.set_ylim(img_h,0)
    ax.set_aspect('equal', adjustable='box')
    if hide_axis:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=3)


def draw_dets(ax, det: np.ndarray, color='0.25', alpha=0.25, s=6, label=None):
    if det.size == 0: return
    c=boxes_to_centers(det)
    ax.scatter(c[:,0], c[:,1], s=s, c=color, alpha=alpha, linewidths=0, label=label)


def draw_tracks(ax, arr: np.ndarray, color='tab:blue', lw=1.2, alpha=0.75, s=8, label=None, max_tracks: Optional[int]=None):
    if arr.size == 0: return
    ids=np.unique(arr[:,1].astype(int))
    if max_tracks is not None:
        ids=ids[:max_tracks]
    first=True
    for oid in ids:
        tr=arr[arr[:,1].astype(int)==oid]
        tr=tr[np.argsort(tr[:,0])]
        c=boxes_to_centers(tr)
        if len(c)==1:
            ax.scatter(c[:,0],c[:,1],s=s,c=color,alpha=alpha,label=label if first else None)
        else:
            ax.plot(c[:,0],c[:,1],color=color,lw=lw,alpha=alpha,label=label if first else None)
            ax.scatter(c[0,0],c[0,1],s=s*1.2,c=color,alpha=alpha,marker='o')
            ax.scatter(c[-1,0],c[-1,1],s=s*1.2,c=color,alpha=alpha,marker='^')
        first=False


def save_overview_4panel(path: str|Path, det: np.ndarray, gt: np.ndarray, pred: np.ndarray, img_w=256, img_h=256, start=None, end=None, roi=None, title=''):
    det=filter_roi(filter_frames(det,start,end),roi)
    gt=filter_roi(filter_frames(gt,start,end),roi)
    pred=filter_roi(filter_frames(pred,start,end),roi)
    fig,axs=plt.subplots(1,4,figsize=(12,3.2),dpi=220)
    panels=[('Detections',0),('GT',1),('Prediction',2),('Overlay',3)]
    for t,i in panels:
        setup_ax(axs[i],img_w,img_h,t,roi=roi)
    draw_dets(axs[0],det,color='0.15',alpha=0.28,s=5)
    draw_tracks(axs[1],gt,color='red',lw=1.5,alpha=0.78,s=8)
    draw_tracks(axs[2],pred,color='blue',lw=1.2,alpha=0.76,s=7)
    draw_dets(axs[3],det,color='0.55',alpha=0.18,s=4)
    draw_tracks(axs[3],gt,color='red',lw=1.7,alpha=0.65,s=7,label='GT')
    draw_tracks(axs[3],pred,color='blue',lw=1.15,alpha=0.70,s=6,label='Pred')
    if title:
        fig.suptitle(title,fontsize=11,y=1.02)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=0.6)
    fig.savefig(path,bbox_inches='tight')
    plt.close(fig)


def save_comparison_grid(path: str|Path, det: np.ndarray, gt: np.ndarray, preds: Dict[str,np.ndarray], img_w=256, img_h=256, start=None, end=None, roi=None, max_cols=3, title=''):
    det_f=filter_roi(filter_frames(det,start,end),roi)
    gt_f=filter_roi(filter_frames(gt,start,end),roi)
    names=['Noisy Det.','GT']+list(preds.keys())
    n=len(names); cols=max_cols; rows=int(np.ceil(n/cols))
    fig,axs=plt.subplots(rows,cols,figsize=(cols*3.0,rows*3.0),dpi=240)
    axs=np.array(axs).reshape(-1)
    for ax in axs[n:]: ax.axis('off')
    for idx,name in enumerate(names):
        ax=axs[idx]; setup_ax(ax,img_w,img_h,name,roi=roi)
        if name=='Noisy Det.':
            draw_dets(ax,det_f,color='0.15',alpha=0.28,s=5)
        elif name=='GT':
            draw_tracks(ax,gt_f,color='red',lw=1.5,alpha=0.78,s=8)
        else:
            pred=filter_roi(filter_frames(preds[name],start,end),roi)
            draw_dets(ax,det_f,color='0.6',alpha=0.12,s=3)
            draw_tracks(ax,gt_f,color='red',lw=1.1,alpha=0.35,s=5)
            draw_tracks(ax,pred,color='blue',lw=1.2,alpha=0.78,s=6)
    if title: fig.suptitle(title,fontsize=11,y=1.01)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(path,bbox_inches='tight')
    plt.close(fig)


def save_3d(path: str|Path, gt: np.ndarray, pred: np.ndarray, det: Optional[np.ndarray]=None, start=None, end=None, roi=None, title=''):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    gt=filter_roi(filter_frames(gt,start,end),roi)
    pred=filter_roi(filter_frames(pred,start,end),roi)
    det=filter_roi(filter_frames(det,start,end),roi) if det is not None else None
    fig=plt.figure(figsize=(6,5),dpi=220)
    ax=fig.add_subplot(111,projection='3d')
    if det is not None and det.size:
        c=boxes_to_centers(det); ax.scatter(c[:,0],c[:,1],det[:,0],s=2,c='0.65',alpha=0.15)
    for arr,color,lw,alpha,label in [(gt,'red',1.3,0.75,'GT'),(pred,'blue',1.0,0.75,'Pred')]:
        ids=np.unique(arr[:,1].astype(int)) if arr.size else []
        first=True
        for oid in ids:
            tr=arr[arr[:,1].astype(int)==oid]; tr=tr[np.argsort(tr[:,0])]
            c=boxes_to_centers(tr)
            ax.plot(c[:,0],c[:,1],tr[:,0],color=color,lw=lw,alpha=alpha,label=label if first else None)
            first=False
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('frame')
    if title: ax.set_title(title,fontsize=10)
    if len(ax.get_legend_handles_labels()[0])>0: ax.legend(fontsize=8)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout()
    fig.savefig(path,bbox_inches='tight')
    plt.close(fig)
