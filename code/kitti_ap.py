"""KITTI 3D AP / recall with ignore masks, built on OpenPCDet's own evaluator (OpenPCDet source is not modified).

Restricting an evaluation to a subset of objects (a distance band, a points-per-object bin) uses the same mechanism
KITTI uses for difficulty levels:
  * ground truth outside the subset gets occluded=3 (> every difficulty's limit), so it is "ignored": it is neither a
    miss nor does a detection matched to it count as a false positive;
  * detections outside the subset get a zero-height 2D box, so KITTI's minimum-height rule ignores them: an unmatched
    one is not a false positive.
Only 3D AP at the official thresholds (Car 0.7, Pedestrian 0.5, Cyclist 0.5) is computed, which is ~6x cheaper than
the full report. The 3D IoU matrix does not depend on the masks, so it is computed once and reused.
"""
import contextlib
import numpy as np
try:
    import torch  # noqa: F401  must come before pcdet's compiled ops (libc10); absent only in torch-free unit tests
except ImportError:
    pass
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

CLASSES = ('Car', 'Pedestrian', 'Cyclist')
CLASS_ID = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
IOU_3D = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
DIFFICULTIES = ('easy', 'moderate', 'hard')


def distance(anno):
    """Horizontal distance (m) from the sensor, from camera-frame locations (x right, z forward)."""
    loc = np.asarray(anno['location'], dtype=np.float64).reshape(-1, 3)
    return np.hypot(loc[:, 0], loc[:, 2])


def mask_gt(gts, keep_fn):
    """keep_fn(anno) -> bool array over that frame's ground-truth objects. Others become ignored."""
    out = []
    for a in gts:
        a = dict(a)
        keep = np.asarray(keep_fn(a), dtype=bool)
        occ = np.array(a['occluded'], dtype=np.float64)
        occ[~keep] = 3
        a['occluded'] = occ
        out.append(a)
    return out


def mask_dt(dts, keep_fn):
    out = []
    for a in dts:
        a = dict(a)
        keep = np.asarray(keep_fn(a), dtype=bool)
        bbox = np.array(a['bbox'], dtype=np.float64)
        bbox[~keep, 3] = bbox[~keep, 1]          # zero height -> ignored by KITTI's minimum-height rule
        a['bbox'] = bbox
        out.append(a)
    return out


@contextlib.contextmanager
def cached_overlaps():
    """Compute the IoU matrices once and reuse them for every masked evaluation inside the block.
    All evaluations inside one block must use the same detections and ground-truth geometry (only the masks change)."""
    original = kitti_eval.calculate_iou_partly
    store = {}

    def cached(a, b, metric, num_parts=50):
        if metric not in store:
            store[metric] = original(a, b, metric, num_parts)
        return store[metric]

    kitti_eval.calculate_iou_partly = cached
    try:
        yield
    finally:
        kitti_eval.calculate_iou_partly = original


def ap3d(gts, dts, classes=CLASSES):
    """3D AP at the official thresholds. Returns {class: {'R11': {'easy':..,'moderate':..,'hard':..}, 'R40': {...}}}."""
    ids = [CLASS_ID[c] for c in classes]
    min_overlaps = np.zeros((1, 3, len(ids)))
    for k, c in enumerate(classes):
        min_overlaps[0, :, k] = IOU_3D[c]
    ret = kitti_eval.eval_class(gts, dts, ids, [0, 1, 2], 2, min_overlaps, False)
    prec = ret['precision'][:, :, 0, :]                    # [class, difficulty, 41 points]
    r11, r40 = kitti_eval.get_mAP(prec), kitti_eval.get_mAP_R40(prec)
    return {c: {'R11': dict(zip(DIFFICULTIES, map(float, r11[k]))),
                'R40': dict(zip(DIFFICULTIES, map(float, r40[k])))} for k, c in enumerate(classes)}


def moderate_valid(anno, cls):
    """Ground truth of class `cls` that KITTI's Moderate difficulty counts (same rules as eval.clean_data)."""
    name = np.asarray(anno['name'])
    h = np.asarray(anno['bbox'])[:, 3] - np.asarray(anno['bbox'])[:, 1]
    return ((name == cls) & (np.asarray(anno['occluded']) <= 1) & (np.asarray(anno['truncated']) <= 0.3) & (h > 25))


def gt_recall_table(gts, dts, classes=CLASSES, min_score=0.0):
    """One row per Moderate-valid ground-truth object: class, distance, points inside, best 3D IoU with a same-class
    detection (score >= min_score), and whether it counts as found at the official IoU. Used for recall by distance
    and by points-per-object, which AP cannot split cleanly (false positives do not belong to a bin)."""
    rows = {'cls': [], 'dist': [], 'npts': [], 'iou': [], 'frame': []}
    for fi, (g, d) in enumerate(zip(gts, dts)):
        for cls in classes:
            sel = np.where(moderate_valid(g, cls))[0]
            if len(sel) == 0:
                continue
            gb = np.concatenate([np.asarray(g['location'])[sel], np.asarray(g['dimensions'])[sel],
                                 np.asarray(g['rotation_y'])[sel][:, None]], 1).astype(np.float64)
            dsel = np.where((np.asarray(d['name']) == cls) & (np.asarray(d['score']) >= min_score))[0]
            if len(dsel):
                db = np.concatenate([np.asarray(d['location'])[dsel], np.asarray(d['dimensions'])[dsel],
                                     np.asarray(d['rotation_y'])[dsel][:, None]], 1).astype(np.float64)
                best = kitti_eval.d3_box_overlap(gb, db).max(axis=1)
            else:
                best = np.zeros(len(sel))
            npts = np.asarray(g.get('num_points_in_gt', -np.ones(len(g['name']))))[sel]
            rows['cls'] += [cls] * len(sel)
            rows['dist'] += distance(g)[sel].tolist()
            rows['npts'] += npts.tolist()
            rows['iou'] += best.tolist()
            rows['frame'] += [fi] * len(sel)
    return {k: np.asarray(v) for k, v in rows.items()}
