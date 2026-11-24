"""Run the pretrained OpenPCDet detector on perturbed KITTI val point clouds.

For every condition in perturb.CONDITIONS it saves, under <out>/<condition>/:
  result.pkl   detections in KITTI annotation format (same layout test.py writes)
  timing.json  per-frame model latency (explicit CUDA sync, batch size 1), CPU preprocessing time, peak GPU memory,
               points and pillars per frame
Finished conditions are skipped, so a Colab disconnect only costs the condition that was running.

  python run_conditions.py --ckpt /content/ckpt/pointpillar_7728.pth --out /content/drive/MyDrive/pcdet_results/robustness
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

os.chdir(Path(__file__).resolve().parent)          # OpenPCDet configs use paths relative to tools/
import perturb
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class PerturbedKitti(KittiDataset):
    """KittiDataset whose raw point clouds are perturbed before the camera-FOV crop and voxelization."""

    def __init__(self, *args, ops=None, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ops, self.seed = ops or [], seed

    def get_lidar(self, idx):
        points = super().get_lidar(idx)
        if not self.ops:
            return points
        return perturb.apply(points, self.ops, np.random.default_rng([self.seed, int(idx)]))


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def make_dataset(ops, args, logger):
    ds = PerturbedKitti(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, root_path=None, training=False,
                        logger=logger, ops=ops, seed=args.seed)
    if args.max_frames:
        ds.kitti_infos = ds.kitti_infos[:args.max_frames]
    return ds


def run_condition(name, model, args, logger, out_dir):
    ds = make_dataset(perturb.CONDITIONS[name], args, logger)

    # CPU preprocessing per frame in the main process: read .bin + perturb + FOV crop + voxelization
    n_prep = min(len(ds), 105)
    for i in range(min(5, n_prep)):
        ds[i]
    t0 = time.perf_counter()
    for i in range(5, n_prep):
        ds[i]
    prep_ms = (time.perf_counter() - t0) / max(n_prep - 5, 1) * 1000

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True,
                        collate_fn=ds.collate_batch, drop_last=False)
    det_annos, times, n_points, n_pillars = [], [], [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for batch in tqdm.tqdm(loader, desc=name, dynamic_ncols=True):
        load_data_to_gpu(batch)
        n_points.append(int(batch['points'].shape[0]))
        n_pillars.append(int(batch['voxels'].shape[0]))
        cuda_sync()
        t = time.perf_counter()
        with torch.no_grad():
            pred_dicts, _ = model(batch)
        cuda_sync()
        times.append((time.perf_counter() - t) * 1000)
        det_annos += ds.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)

    steady = np.array(times[args.warmup:] if len(times) > args.warmup else times)
    timing = {
        'condition': name, 'description': perturb.DESCRIPTIONS.get(name, ''), 'frames': len(times), 'warmup_frames': args.warmup,
        'forward_ms_mean': float(steady.mean()), 'forward_ms_median': float(np.median(steady)),
        'forward_ms_p95': float(np.percentile(steady, 95)), 'fps_from_mean': float(1000.0 / steady.mean()),
        'prep_ms_mean': float(prep_ms),
        'peak_gpu_mem_mb': float(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else None,
        'points_per_frame_mean': float(np.mean(n_points)), 'pillars_per_frame_mean': float(np.mean(n_pillars)),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'times_ms': [round(t, 3) for t in times],
    }
    cond_dir = out_dir / name
    cond_dir.mkdir(parents=True, exist_ok=True)
    with open(cond_dir / 'timing.json', 'w') as f:
        json.dump(timing, f)
    with open(cond_dir / 'result.pkl', 'wb') as f:          # written last: its presence means the condition is complete
        pickle.dump(det_annos, f)
    logger.info('%s: %d frames, forward %.1f ms (median %.1f, p95 %.1f), prep %.1f ms, %.0f points, %.0f pillars',
                name, len(det_annos), timing['forward_ms_mean'], timing['forward_ms_median'], timing['forward_ms_p95'],
                prep_ms, timing['points_per_frame_mean'], timing['pillars_per_frame_mean'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg_file', default='cfgs/kitti_models/pointpillar.yaml')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--conditions', nargs='+', default=list(perturb.CONDITIONS))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_frames', type=int, default=None, help='first N val frames only (quick test)')
    p.add_argument('--warmup', type=int, default=20, help='frames excluded from the latency statistics')
    p.add_argument('--force', action='store_true', help='recompute conditions that already have results')
    args = p.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=make_dataset([], args, logger))
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    for name in args.conditions:
        if (out_dir / name / 'result.pkl').exists() and not args.force:
            logger.info('%s: already done, skipping (use --force to redo)', name)
            continue
        run_condition(name, model, args, logger, out_dir)


if __name__ == '__main__':
    main()
