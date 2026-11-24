import pickle, yaml
from pathlib import Path
from easydict import EasyDict
import torch  # first, so the compiled ops can find libc10
from pcdet.datasets.kitti.kitti_dataset import KittiDataset

root = Path('/content/OpenPCDet')
data = root / 'data' / 'kitti'
cfg = EasyDict(yaml.safe_load(open(root / 'tools/cfgs/dataset_configs/kitti_dataset.yaml')))
ds = KittiDataset(dataset_cfg=cfg, class_names=['Car', 'Pedestrian', 'Cyclist'], root_path=data, training=False)

infos = {}
for split in ('train', 'val'):
    ds.set_split(split)
    infos[split] = ds.get_infos(num_workers=4, has_label=True, count_inside_pts=True)
    with open(data / f'kitti_infos_{split}.pkl', 'wb') as f:
        pickle.dump(infos[split], f)
    print(split, len(infos[split]), 'frames')
with open(data / 'kitti_infos_trainval.pkl', 'wb') as f:
    pickle.dump(infos['train'] + infos['val'], f)
