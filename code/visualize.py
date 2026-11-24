"""Bird's-eye-view and 3D figures: point cloud + ground truth + predictions, for any frame and condition.

  from visualize import render
  render('000123', 'clean', infos, results_dir, kitti_root, 'out/figs')              # one condition
  render('000123', ['clean', 'drop75'], ...)                                         # before/after side by side
Ground truth: solid green. Predictions: dashed, coloured by class, labelled with score (threshold --min_score).
Boxes are in the LiDAR frame (x forward, y left, z up), as OpenPCDet's `gt_boxes_lidar` / `boxes_lidar`.
"""
import os

os.environ["MPLBACKEND"] = "Agg"   # Colab sets an inline backend that a "!python script.py" subprocess cannot use
import pickle
from pathlib import Path

import numpy as np

import perturb

CLASS_COLOR = {'Car': '#eb6834', 'Pedestrian': '#6250d6', 'Cyclist': '#1baf7a'}
GT_COLOR = '#008300'
XLIM, YLIM = (0, 70.4), (-40, 40)


def box_corners(b):
    """b = (x, y, z, dx, dy, dz, heading), z at the box centre -> 8x3 corners (bottom 0-3, top 4-7)."""
    x, y, z, dx, dy, dz, h = [float(v) for v in b[:7]]
    c = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]]) / 2
    R = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
    xy = c @ R.T + [x, y]
    return np.vstack([np.c_[xy, np.full(4, z - dz / 2)], np.c_[xy, np.full(4, z + dz / 2)]])


def load_points(kitti_root, frame_id, condition, seed=0):
    f = Path(kitti_root) / 'training' / 'velodyne' / f'{frame_id}.bin'
    pts = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    pts = perturb.perturb_frame(pts, condition, frame_id, seed)
    m = (pts[:, 0] > XLIM[0]) & (pts[:, 0] < XLIM[1]) & (np.abs(pts[:, 1]) < YLIM[1])
    return pts[m]


def frame_boxes(infos, dets, frame_id, min_score):
    info = next(i for i in infos if i['point_cloud']['lidar_idx'] == frame_id)
    gt = np.asarray(info['annos']['gt_boxes_lidar'])
    names = np.asarray(info['annos']['name'])
    gt_names = names[names != 'DontCare']          # gt_boxes_lidar has no DontCare rows, so align the names with it
    assert len(gt_names) == len(gt), (len(gt_names), len(gt))
    ok = np.isin(gt_names, list(CLASS_COLOR))       # draw only the three evaluated classes (not Van, Truck, Misc, ...)
    d = next(x for x in dets if x['frame_id'] == frame_id)
    keep = np.asarray(d['score']) >= min_score
    return gt[ok], gt_names[ok], np.asarray(d['boxes_lidar'])[keep], np.asarray(d['name'])[keep], np.asarray(d['score'])[keep]


def _bev(ax, pts, gt, gt_n, pr, pr_n, pr_s):
    ax.scatter(pts[:, 0], pts[:, 1], s=0.4, c=np.clip(pts[:, 2], -2, 1.5), cmap='Greys', vmin=-2.5, vmax=2, linewidths=0)
    for b in gt:
        c = box_corners(b)[:4]
        ax.plot(*np.r_[c, c[:1]].T[:2], color=GT_COLOR, lw=1.6)
    for b, n, s in zip(pr, pr_n, pr_s):
        c = box_corners(b)[:4]
        ax.plot(*np.r_[c, c[:1]].T[:2], color=CLASS_COLOR.get(n, 'k'), lw=1.4, ls='--')
        ax.text(c[:, 0].max(), c[:, 1].max(), f'{s:.2f}', fontsize=6, color=CLASS_COLOR.get(n, 'k'))
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect('equal')
    ax.set_xlabel('x forward (m)'); ax.set_ylabel('y left (m)')


EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _view3d(ax, pts, gt, pr, pr_n, elev=22, azim=180):
    sel = (pts[:, 0] < 40) & (np.abs(pts[:, 1]) < 20)
    ax.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 2], s=0.3, c=pts[sel, 2], cmap='viridis', vmin=-2, vmax=1.5, linewidths=0)
    for b in gt:
        c = box_corners(b)
        for i, j in EDGES:
            ax.plot(*c[[i, j]].T, color=GT_COLOR, lw=1.2)
    for b, n in zip(pr, pr_n):
        c = box_corners(b)
        for i, j in EDGES:
            ax.plot(*c[[i, j]].T, color=CLASS_COLOR.get(n, 'k'), lw=1.0, ls='--')
    ax.set_xlim(0, 40); ax.set_ylim(-20, 20); ax.set_zlim(-3, 3)
    ax.set_box_aspect((40, 40, 8))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')


def render(frame_id, conditions, infos, results_dir, kitti_root, out_dir, min_score=0.3, seed=0, three_d=True):
    """Save <out_dir>/<frame>_bev.png and (optionally) <frame>_3d.png; one column per condition."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions = [conditions] if isinstance(conditions, str) else list(conditions)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for kind in (['bev', '3d'] if three_d else ['bev']):
        n = len(conditions)
        fig = plt.figure(figsize=(7.5 * n if kind == 'bev' else 6.5 * n, 5.2 if kind == 'bev' else 5.5))
        for k, cond in enumerate(conditions):
            with open(Path(results_dir) / cond / 'result.pkl', 'rb') as f:
                dets = pickle.load(f)
            pts = load_points(kitti_root, frame_id, cond, seed)
            gt, gt_n, pr, pr_n, pr_s = frame_boxes(infos, dets, frame_id, min_score)
            if kind == 'bev':
                ax = fig.add_subplot(1, n, k + 1)
                _bev(ax, pts, gt, gt_n, pr, pr_n, pr_s)
            else:
                ax = fig.add_subplot(1, n, k + 1, projection='3d')
                _view3d(ax, pts, gt, pr, pr_n)
            ax.set_title(f'{frame_id}  {cond}  ({len(pts):,} pts in view, {len(pr)} detections, {len(gt)} GT)', fontsize=9)
        handles = [plt.Line2D([], [], color=GT_COLOR, lw=1.6, label='ground truth')] + \
                  [plt.Line2D([], [], color=c, ls='--', lw=1.4, label=f'pred {n}') for n, c in CLASS_COLOR.items()]
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        path = out_dir / f'{frame_id}_{kind}_{"-".join(conditions)}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)
    return saved


if __name__ == '__main__':
    import argparse
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument('--infos', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--kitti', required=True, help='KITTI root that contains training/velodyne')
    p.add_argument('--out', required=True)
    p.add_argument('--frames', nargs='*', help='frame ids; default: picked from failure_frames.csv in --analysis')
    p.add_argument('--analysis', help='folder written by analyze.py')
    p.add_argument('--conditions', nargs='+', default=['clean', 'drop75'])
    p.add_argument('--min_score', type=float, default=0.3)
    a = p.parse_args()
    with open(a.infos, 'rb') as f:
        infos = pickle.load(f)
    frames = a.frames
    if not frames:
        ff = pd.read_csv(Path(a.analysis) / 'failure_frames.csv', dtype={'frame_id': str})
        frames = list(dict.fromkeys(ff[ff.kind == 'clean_misses'].frame_id[:2].tolist() +
                                    ff[(ff.kind == 'biggest_loss') & (ff.condition == a.conditions[-1])].frame_id[:2].tolist()))
    for fid in frames:
        for path in render(fid, a.conditions, infos, a.results, a.kitti, a.out, a.min_score):
            print(path)
