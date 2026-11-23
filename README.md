# Lidar-based-object-detection

LiDAR-only 3D object detection with **PointPillars** (OpenPCDet) on the KITTI dataset, run on a Google Colab Tesla T4. The project evaluates a pretrained checkpoint, tests its robustness to degraded point clouds, and fine-tunes it on degraded data.

## What is in this project

1. **Baseline:** pretrained `pointpillar_7728.pth` on the 3,769 KITTI validation frames.
   Moderate 3D AP (R40): Car 78.4, Pedestrian 51.4, Cyclist 62.9. Car R11 is 77.28, matching the OpenPCDet model zoo.
   Median forward time 45.1 ms per frame (batch size 1).
2. **Robustness study:** seven input conditions (clean, 50% and 75% point dropout, half the beams, 0.1 m and 0.3 m range noise, 20% clutter), with AP by condition and by distance, recall by points per object, and runtime.
3. **Fine-tuning on degraded clouds:** 8 epochs (about 59 minutes on a T4) on randomly corrupted training frames. It recovers 8 to 25 AP points on the worst conditions and costs 1 to 7 points on clean data. This is a single run with no clean-data control.
4. **Qualitative results:** bird's-eye and 3D views with ground truth and predictions, before and after fine-tuning, and a failure-case analysis (range, sparsity, occlusion).
