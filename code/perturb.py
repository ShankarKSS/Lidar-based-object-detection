"""Point-cloud perturbations for the robustness study (plain numpy; no OpenPCDet imports).

Each op takes (points, rng, **params) and returns new points. Points are KITTI velodyne rows (x, y, z, intensity) in
the LiDAR frame (x forward, y left, z up, sensor at the origin). A frame's random numbers come from
np.random.default_rng([seed, frame_number]), so every run, worker and visualization sees the same perturbed frame.
"""
import numpy as np

# Nominal Velodyne HDL-64E vertical layout (degrees): upper block 32 beams ~1/3 deg apart, lower block 32 beams ~1/2 deg.
# KITTI .bin files carry no ring index, so beams are recovered from elevation angle, which makes `beam_downsample`
# an approximation of dropping every other physical beam.
HDL64_ANGLES = np.concatenate([np.linspace(2.0, -8.33, 32), np.linspace(-8.83, -24.33, 32)])   # descending


def random_drop(points, rng, p):
    """Drop each point independently with probability p (cheaper/sparser sensor, or occlusion by rain/dust)."""
    return points[rng.random(len(points)) >= p]


def beam_downsample(points, rng, keep_every=2):
    """Keep every `keep_every`-th beam (2 -> roughly a 32-beam sensor), beams assigned by nearest nominal elevation."""
    elev = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    asc = HDL64_ANGLES[::-1]                                             # ascending for searchsorted
    j = np.clip(np.searchsorted(asc, elev), 1, len(asc) - 1)
    nearest = np.where(np.abs(asc[j] - elev) < np.abs(asc[j - 1] - elev), j, j - 1)
    beam = len(asc) - 1 - nearest                                        # 0 = top beam
    return points[beam % keep_every == 0]


def range_noise(points, rng, sigma):
    """Gaussian noise (std sigma metres) added along each point's ray, as in range-measurement noise."""
    out = points.copy()
    r = np.linalg.norm(out[:, :3], axis=1, keepdims=True)
    r_safe = np.maximum(r, 1e-6)
    out[:, :3] *= (r + rng.normal(0.0, sigma, r.shape)) / r_safe
    return out


def clutter(points, rng, frac, r_range=(3.0, 40.0), z_range=(-1.7, 0.5), half_fov_deg=40.0):
    """Add spurious returns (rain, snow, dust): `frac` x the number of points in the forward camera sector, scattered
    uniformly in range and height inside that sector, with low intensity."""
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    n_forward = int(np.sum((np.abs(az) < half_fov_deg) & (points[:, 0] > 0)))
    n = int(round(frac * n_forward))
    r = rng.uniform(*r_range, n)
    a = np.radians(rng.uniform(-half_fov_deg, half_fov_deg, n))
    extra = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(*z_range, n), rng.uniform(0.0, 0.3, n)], axis=1)
    return np.concatenate([points, extra.astype(points.dtype)], axis=0)


OPS = {'random_drop': random_drop, 'beam_downsample': beam_downsample, 'range_noise': range_noise, 'clutter': clutter}

# name -> list of (op, params); applied in order. Edit or add entries freely.
CONDITIONS = {
    'clean':      [],
    'drop50':     [('random_drop', {'p': 0.5})],
    'drop75':     [('random_drop', {'p': 0.75})],
    'beams_half': [('beam_downsample', {'keep_every': 2})],
    'noise_0.1m': [('range_noise', {'sigma': 0.1})],
    'noise_0.3m': [('range_noise', {'sigma': 0.3})],
    'clutter_20': [('clutter', {'frac': 0.2})],
}
DESCRIPTIONS = {
    'clean': 'unmodified', 'drop50': 'random 50% of points removed', 'drop75': 'random 75% of points removed',
    'beams_half': 'every other beam (~32-beam sensor, approximate)', 'noise_0.1m': 'range noise sigma 0.1 m',
    'noise_0.3m': 'range noise sigma 0.3 m', 'clutter_20': 'spurious returns, +20% points in the forward sector',
}


def apply(points, ops, rng):
    for name, params in ops:
        points = OPS[name](points, rng, **params)
    return np.ascontiguousarray(points, dtype=np.float32)


def perturb_frame(points, condition, frame_id, seed=0):
    """Deterministically perturb one frame (frame_id like '000123')."""
    return apply(points, CONDITIONS[condition], np.random.default_rng([seed, int(frame_id)]))


# ---------------------------------------------------------------------------------------------------------------------
# Random corruptions for fine-tuning (continuous severities, so the evaluation levels above are not literally the
# training levels, but the corruption types are the same unless a type is held out with `hold_out`).
TRAIN_OPS = ('random_drop', 'beam_downsample', 'range_noise', 'clutter')


def sample_training_ops(rng, p_clean=0.2, p_two=0.3, hold_out=()):
    """Random list of (op, params) for one training frame. With prob p_clean: no corruption. Otherwise one corruption,
    or (prob p_two) two different ones. Ops named in `hold_out` are never used, to test generalization to an unseen type."""
    if rng.random() < p_clean:
        return []
    names = [n for n in TRAIN_OPS if n not in hold_out]
    k = 2 if (len(names) > 1 and rng.random() < p_two) else 1
    chosen = list(rng.choice(names, size=k, replace=False))
    params = {'random_drop': lambda: {'p': float(rng.uniform(0.2, 0.85))},
              'beam_downsample': lambda: {'keep_every': 2},
              'range_noise': lambda: {'sigma': float(rng.uniform(0.05, 0.35))},
              'clutter': lambda: {'frac': float(rng.uniform(0.05, 0.3))}}
    return [(n, params[n]()) for n in chosen]
