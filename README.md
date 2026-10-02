# Master's Thesis

Repository for Master's Thesis project: *Machine learning methods for removing redundant lidar data*.

## Scripts

- [scripts/data_saver.py](scripts/data_saver.py): Real-time ROS node for synchronizing and recording raw and RMS-filtered point clouds.
- [scripts/data_saver_old.py](scripts/data_saver_old.py): Reference / backup data saving script.
- [scripts/extract_labels.py](scripts/extract_labels.py): Offline label extraction applying **Gaussian Target Smearing** (Equation 5.1, $\sigma = 0.10\text{ m}$) to produce continuous regression heatmaps for neural network training.
