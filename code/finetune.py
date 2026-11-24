"""Fine-tune a pretrained OpenPCDet model on randomly degraded point clouds, using OpenPCDet's own train.py unchanged.

This launcher only (1) patches the training-time DataProcessor so each training frame's points get a random corruption
from perturb.sample_training_ops (after OpenPCDet's own augmentation: GT sampling, flip, rotation, scaling; before
range masking and voxelization), then (2) runs train.main(). Everything else on the command line goes to train.py.

  python finetune.py --hold_out clutter --cfg_file cfgs/kitti_models/pointpillar.yaml --pretrained_model ckpt.pth \
      --epochs 8 --extra_tag finetune_degraded --set OPTIMIZATION.LR 0.0003

If KITTI's road-plane files (training/planes) are absent, USE_ROAD_PLANE is switched off for GT sampling.

Evaluation frames are corrupted before the camera-FOV crop (run_conditions.py); training frames are corrupted after
the FOV crop and augmentation. For per-point corruptions (drop, noise, beams) that is equivalent; for clutter the
spurious points are placed in a +-40 deg sector of the augmented (rotated/flipped) cloud rather than of the raw one.
"""
import argparse
import sys

import numpy as np
import torch  # before pcdet's compiled ops

import perturb


def install(p_clean, hold_out):
    from pcdet.datasets.processor.data_processor import DataProcessor
    original = DataProcessor.forward

    def forward(self, data_dict):
        if self.training and data_dict.get('points') is not None:
            ops = perturb.sample_training_ops(np.random.default_rng(), p_clean=p_clean, hold_out=hold_out)
            if ops:
                data_dict['points'] = perturb.apply(data_dict['points'], ops, np.random.default_rng())
        return original(self, data_dict)

    DataProcessor.forward = forward


if __name__ == '__main__':
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--p_clean', type=float, default=0.2, help='fraction of training frames left uncorrupted')
    ap.add_argument('--hold_out', nargs='*', default=[], choices=perturb.TRAIN_OPS,
                    help='corruption types never used in training (to test generalization)')
    own, rest = ap.parse_known_args()
    install(own.p_clean, tuple(own.hold_out))
    sys.argv = [sys.argv[0]] + rest
    import train
    from pathlib import Path

    original_parse_config = train.parse_config

    def parse_config_without_missing_planes():
        args, cfg = original_parse_config()
        planes = Path(cfg.DATA_CONFIG.DATA_PATH) / 'training' / 'planes'
        if not planes.exists():          # road planes are a separate KITTI download; without them GT sampling cannot snap to the road
            for aug in cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST:
                if aug.get('USE_ROAD_PLANE', False):
                    aug['USE_ROAD_PLANE'] = False
                    print(f'finetune.py: {planes} not found, USE_ROAD_PLANE disabled for {aug["NAME"]}')
        return args, cfg

    train.parse_config = parse_config_without_missing_planes
    train.main()
