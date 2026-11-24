"""Side-by-side pretrained vs fine-tuned results from two analyze.py output folders.

  python compare.py --a <analysis dir, pretrained> --b <analysis dir, fine-tuned> --out <dir> [--names pretrained finetuned]
Writes compare_ap.csv (Moderate 3D AP and difference per class and condition), compare_distance.csv, chart_compare.png.
"""
import os

os.environ["MPLBACKEND"] = "Agg"   # Colab sets an inline backend that a "!python script.py" subprocess cannot use
import argparse
from pathlib import Path

import pandas as pd

INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
SERIES = ['#2a78d6', '#eb6834']          # categorical slots 1 and 2


def merge(a, b, keys, value, names):
    m = a.merge(b, on=keys, suffixes=('_a', '_b'))
    out = m[keys].copy()
    out[names[0]] = m[value + '_a']
    out[names[1]] = m[value + '_b']
    out['change'] = out[names[1]] - out[names[0]]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--names', nargs=2, default=['pretrained', 'finetuned'])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B = Path(args.a), Path(args.b)
    ap = merge(pd.read_csv(A / 'ap_overall.csv'), pd.read_csv(B / 'ap_overall.csv'), ['condition', 'class'], 'moderate', args.names)
    dist = merge(pd.read_csv(A / 'ap_by_distance.csv'), pd.read_csv(B / 'ap_by_distance.csv'),
                 ['condition', 'class', 'band'], 'moderate_AP', args.names)
    ap.to_csv(out / 'compare_ap.csv', index=False)
    dist.to_csv(out / 'compare_distance.csv', index=False)

    pd.set_option('display.width', 200, 'display.float_format', '{:.1f}'.format)
    for cls in ('Car', 'Pedestrian', 'Cyclist'):
        print(f'\n{cls}: Moderate 3D AP (R40, %)\n', ap[ap['class'] == cls].drop(columns='class').set_index('condition'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    conds = list(ap['condition'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True, facecolor=SURF)
    for ax, cls in zip(axes, ('Car', 'Pedestrian', 'Cyclist')):
        d = ap[ap['class'] == cls].set_index('condition').loc[conds]
        w = 0.38
        for k, n in enumerate(args.names):
            ax.bar(np.arange(len(conds)) + (k - 0.5) * (w + 0.02), d[n], width=w, color=SERIES[k], label=n)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, rotation=40, ha='right', color=INK2, fontsize=9)
        ax.set_title(cls, color=INK, loc='left', fontsize=12)
        ax.set_facecolor(SURF)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('3D AP, Moderate (R40, %)', color=INK2)
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.suptitle('Before vs after fine-tuning on degraded point clouds', color=INK, x=0.01, ha='left', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / 'chart_compare.png', dpi=160, facecolor=SURF)
    print('\nWrote', out)


if __name__ == '__main__':
    main()
