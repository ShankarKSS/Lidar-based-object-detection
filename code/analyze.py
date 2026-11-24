"""Turn the per-condition detections from run_conditions.py into tables and charts.

  python analyze.py --infos .../kitti_infos_val.pkl --results <out of run_conditions> --out <dir>

Writes into --out: ap_overall.csv, ap_by_distance.csv, recall_by_distance.csv, recall_by_points.csv, runtime.csv,
failure_frames.csv and four PNG charts. All AP values are 3D AP (R40) at the official IoU thresholds
(Car 0.7, Pedestrian/Cyclist 0.5), Moderate difficulty unless a column says otherwise.
"""
import os

os.environ["MPLBACKEND"] = "Agg"   # Colab sets an inline backend that a "!python script.py" subprocess cannot use
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import kitti_ap
import perturb

BANDS = {'0-20 m': (0, 20), '20-40 m': (20, 40), '40-70 m': (40, 70.4)}
POINT_BINS = [(1, 20), (20, 50), (50, 150), (150, 10 ** 9)]
POINT_LABELS = ['<20 pts', '20-49 pts', '50-149 pts', '150+ pts']


def load(infos_path, results_dir, conditions):
    with open(infos_path, 'rb') as f:
        infos = pickle.load(f)
    dets, timing = {}, {}
    for c in conditions:
        d = Path(results_dir) / c
        if not (d / 'result.pkl').exists():
            print('skipping', c, '(no result.pkl)')
            continue
        with open(d / 'result.pkl', 'rb') as f:
            dets[c] = pickle.load(f)
        timing[c] = json.load(open(d / 'timing.json'))
    n = min(len(v) for v in dets.values())
    infos = infos[:n]
    for c in dets:
        dets[c] = dets[c][:n]
        assert [x['frame_id'] for x in dets[c]] == [i['point_cloud']['lidar_idx'] for i in infos], 'frame order mismatch'
    gts = [i['annos'] for i in infos]
    return gts, dets, timing


def ap_tables(gts, dets, key='R40'):
    """Overall AP (all difficulties) and Moderate AP per distance band, per condition."""
    overall, banded = [], []
    for c, dt in dets.items():
        with kitti_ap.cached_overlaps():
            res = kitti_ap.ap3d(gts, dt)
            for cls in kitti_ap.CLASSES:
                row = {'condition': c, 'class': cls}
                row.update({d: res[cls][key][d] for d in kitti_ap.DIFFICULTIES})
                overall.append(row)
            for band, (lo, hi) in BANDS.items():
                keep = lambda a, lo=lo, hi=hi: (kitti_ap.distance(a) >= lo) & (kitti_ap.distance(a) < hi)
                r = kitti_ap.ap3d(kitti_ap.mask_gt(gts, keep), kitti_ap.mask_dt(dt, keep))
                for cls in kitti_ap.CLASSES:
                    n = sum(int(kitti_ap.moderate_valid(g, cls)[keep(g)].sum()) for g in gts)
                    banded.append({'condition': c, 'class': cls, 'band': band, 'moderate_AP': r[cls][key]['moderate'],
                                   'n_gt_moderate': n})
    return pd.DataFrame(overall), pd.DataFrame(banded)


def recall_tables(gts, dets, clean='clean', min_score=0.0):
    """Recall (found at the official IoU) by distance band and by points-per-object bin.
    Points-per-object comes from the CLEAN cloud so every condition uses the same objects and bins."""
    by_dist, by_pts = [], []
    for c, dt in dets.items():
        t = kitti_ap.gt_recall_table(gts, dt, min_score=min_score)
        found = t['iou'] >= np.array([kitti_ap.IOU_3D[k] for k in t['cls']])
        for cls in kitti_ap.CLASSES:
            m = t['cls'] == cls
            for band, (lo, hi) in BANDS.items():
                s = m & (t['dist'] >= lo) & (t['dist'] < hi)
                by_dist.append({'condition': c, 'class': cls, 'band': band, 'n': int(s.sum()),
                                'recall': float(found[s].mean()) if s.any() else np.nan})
            for lab, (lo, hi) in zip(POINT_LABELS, POINT_BINS):
                s = m & (t['npts'] >= lo) & (t['npts'] < hi)
                by_pts.append({'condition': c, 'class': cls, 'points_bin': lab, 'n': int(s.sum()),
                               'recall': float(found[s].mean()) if s.any() else np.nan})
    return pd.DataFrame(by_dist), pd.DataFrame(by_pts)


def failure_frames(gts, dets, top=15):
    """Frames with the most missed objects on clean data, and the frames each condition hurts most."""
    miss = {}
    for c, dt in dets.items():
        t = kitti_ap.gt_recall_table(gts, dt)
        found = t['iou'] >= np.array([kitti_ap.IOU_3D[k] for k in t['cls']])
        counts = np.bincount(t['frame'][~found].astype(int), minlength=len(gts))
        miss[c] = counts
    rows = []
    clean = miss['clean']
    for fi in np.argsort(-clean)[:top]:
        rows.append({'kind': 'clean_misses', 'condition': 'clean', 'frame_id': gts_id(gts, dets, fi), 'misses': int(clean[fi]),
                     'extra_misses_vs_clean': 0})
    for c, counts in miss.items():
        if c == 'clean':
            continue
        extra = counts - clean
        for fi in np.argsort(-extra)[:5]:
            rows.append({'kind': 'biggest_loss', 'condition': c, 'frame_id': gts_id(gts, dets, fi), 'misses': int(counts[fi]),
                         'extra_misses_vs_clean': int(extra[fi])})
    return pd.DataFrame(rows)


def gts_id(gts, dets, fi):
    return dets['clean'][fi]['frame_id']


def runtime_table(timing):
    rows = []
    for c, t in timing.items():
        rows.append({'condition': c, 'forward_ms_mean': t['forward_ms_mean'], 'forward_ms_median': t['forward_ms_median'],
                     'forward_ms_p95': t['forward_ms_p95'], 'fps': t['fps_from_mean'], 'prep_ms_cpu': t['prep_ms_mean'],
                     'points_per_frame': t['points_per_frame_mean'], 'pillars_per_frame': t['pillars_per_frame_mean'],
                     'peak_gpu_mem_mb': t['peak_gpu_mem_mb']})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- charts
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
BLUES = ['#9ec5f4', '#3987e5', '#184f95']          # sequential steps for ordered bands (near -> far)
BLUE = '#2a78d6'


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(overall, banded, runtime, by_pts, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conds = list(overall['condition'].unique())
    classes = list(kitti_ap.CLASSES)

    # 1. Moderate 3D AP per condition, one panel per class
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURF)
    for ax, cls in zip(axes, classes):
        d = overall[overall['class'] == cls].set_index('condition').loc[conds]
        ax.bar(range(len(conds)), d['moderate'], color=BLUE, width=0.6)
        for i, v in enumerate(d['moderate']):
            ax.text(i, v + 1.5, f'{v:.0f}', ha='center', color=INK, fontsize=9)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha='right', color=INK2, fontsize=9)
        ax.set_title(cls, color=INK, loc='left', fontsize=12)
        _style(ax)
    axes[0].set_ylabel('3D AP, Moderate (R40, %)', color=INK2)
    axes[0].set_ylim(0, 105)
    fig.suptitle('Accuracy under point-cloud degradation', color=INK, x=0.01, ha='left', fontsize=13)
    fig.tight_layout()
    fig.savefig(Path(out) / 'chart_ap_by_condition.png', dpi=160, facecolor=SURF)
    plt.close(fig)

    # 2. Moderate AP by distance band (Car) - ordered bands use one hue, light -> dark
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURF)
    for ax, cls in zip(axes, classes):
        d = banded[banded['class'] == cls]
        w = 0.26
        for k, band in enumerate(BANDS):
            dd = d[d['band'] == band].set_index('condition').reindex(conds)
            ax.bar(np.arange(len(conds)) + (k - 1) * (w + 0.02), dd['moderate_AP'], width=w, color=BLUES[k], label=band)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha='right', color=INK2, fontsize=9)
        ax.set_title(cls, color=INK, loc='left', fontsize=12)
        _style(ax)
    axes[0].set_ylabel('3D AP, Moderate (R40, %)', color=INK2)
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=9, title='distance', title_fontsize=9)
    fig.suptitle('Accuracy by distance band (bands with few objects are noisy - see ap_by_distance.csv for counts)',
                 color=INK, x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(out) / 'chart_ap_by_distance.png', dpi=160, facecolor=SURF)
    plt.close(fig)

    # 3. Recall by points per object, Car, clean vs each condition would be crowded -> clean + two harshest
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURF)
    d = by_pts[by_pts['class'] == 'Car']
    w = 0.26
    show = [c for c in ('clean', 'drop75', 'noise_0.3m') if c in conds]
    shades = ['#184f95', '#3987e5', '#9ec5f4']
    for k, c in enumerate(show):
        dd = d[d['condition'] == c].set_index('points_bin').reindex(POINT_LABELS)
        ax.bar(np.arange(len(POINT_LABELS)) + (k - 1) * (w + 0.02), dd['recall'] * 100, width=w, color=shades[k], label=c)
    ax.set_xticks(range(len(POINT_LABELS)))
    ax.set_xticklabels(POINT_LABELS, color=INK2)
    ax.set_ylabel('Car recall (%)', color=INK2)
    ax.set_xlabel('LiDAR points inside the object in the clean scan', color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    ax.set_title('Sparse objects are lost first', color=INK, loc='left', fontsize=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(Path(out) / 'chart_recall_by_points.png', dpi=160, facecolor=SURF)
    plt.close(fig)

    # 4. Runtime per condition (median forward pass, p95 as a tick)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURF)
    r = runtime.set_index('condition').loc[conds]
    ax.bar(range(len(conds)), r['forward_ms_median'], color=BLUE, width=0.6)
    ax.scatter(range(len(conds)), r['forward_ms_p95'], color=INK, marker='_', s=300, zorder=3, label='p95')
    for i, v in enumerate(r['forward_ms_median']):
        ax.text(i, v / 2, f'{v:.1f}', ha='center', color='white', fontsize=9)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=40, ha='right', color=INK2, fontsize=9)
    ax.set_ylabel('Forward pass (ms, batch 1)', color=INK2)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    ax.set_title('Model latency per frame (median, with p95)', color=INK, loc='left', fontsize=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(Path(out) / 'chart_runtime.png', dpi=160, facecolor=SURF)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--infos', required=True)
    p.add_argument('--results', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conditions', nargs='+', default=list(perturb.CONDITIONS))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    gts, dets, timing = load(a.infos, a.results, a.conditions)
    overall, banded = ap_tables(gts, dets)
    rdist, rpts = recall_tables(gts, dets)
    runtime = runtime_table(timing)
    fails = failure_frames(gts, dets)
    for name, df in [('ap_overall', overall), ('ap_by_distance', banded), ('recall_by_distance', rdist),
                     ('recall_by_points', rpts), ('runtime', runtime), ('failure_frames', fails)]:
        df.to_csv(out / f'{name}.csv', index=False)

    pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.float_format', '{:.1f}'.format)
    print('\n3D AP (R40, %), all difficulties\n', overall.pivot(index='condition', columns='class', values='moderate')
          .loc[list(dets)].rename(columns=lambda c: c + ' (Mod)'))
    print('\nModerate AP by distance band\n', banded.pivot_table(index='condition', columns=['class', 'band'],
                                                               values='moderate_AP').loc[list(dets)])
    print('\nObjects per band (Moderate GT)\n', banded[banded.condition == 'clean'].pivot(index='band', columns='class',
                                                                                         values='n_gt_moderate'))
    print('\nCar recall by points per object\n', rpts[rpts['class'] == 'Car'].pivot(index='condition', columns='points_bin',
                                                                                 values='recall').loc[list(dets)][POINT_LABELS] * 100)
    print('\nRuntime\n', runtime.set_index('condition'))
    charts(overall, banded, runtime, rpts, out)
    print('\nWrote tables and charts to', out)


if __name__ == '__main__':
    main()
