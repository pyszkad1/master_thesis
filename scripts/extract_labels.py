#!/usr/bin/env python3

import os
import argparse
import inspect
import numpy as np
from scipy.spatial import cKDTree
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

# Support both SciPy <= 1.5 ('n_jobs') and SciPy >= 1.6 ('workers')
_QUERY_KWARG = {'workers': -1} if 'workers' in inspect.signature(cKDTree.query).parameters else {'n_jobs': -1}

def decode_pointcloud2(msg):
    """Custom pure-Python decoder to replace ROS's pc2.read_points"""
    x_offset = next(f.offset for f in msg.fields if f.name == 'x')
    y_offset = next(f.offset for f in msg.fields if f.name == 'y')
    z_offset = next(f.offset for f in msg.fields if f.name == 'z')
    
    dt = np.dtype({
        'names': ['x', 'y', 'z'],
        'formats': ['<f4', '<f4', '<f4'], 
        'offsets': [x_offset, y_offset, z_offset],
        'itemsize': msg.point_step
    })
    
    cloud_arr = np.frombuffer(msg.data, dtype=dt)
    xyz = np.empty((len(cloud_arr), 3), dtype=np.float32)
    xyz[:, 0] = cloud_arr['x']
    xyz[:, 1] = cloud_arr['y']
    xyz[:, 2] = cloud_arr['z']
    
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return xyz

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract soft continuous labels using Gaussian Target Smearing (Thesis Chapter 5)"
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.10,
        help="Spatial spread parameter sigma in meters for Gaussian Target Smearing (default: 0.10 m)"
    )
    parser.add_argument(
        '--cutoff_mult',
        type=float,
        default=4.0,
        help="Distance cutoff multiplier (search pruned at cutoff_mult * sigma, default: 4.0)"
    )
    parser.add_argument(
        '--bag',
        type=str,
        default=None,
        help="Path to synchronized ROS bag (default: synchronized_subset.bag in script directory)"
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help="Output directory for generated .npy files (default: 'labels_soft' in script directory)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    sigma = args.sigma
    cutoff = args.cutoff_mult * sigma

    base_dir = os.path.dirname(os.path.abspath(__file__))
    bag_path = args.bag if args.bag else os.path.join(base_dir, 'synchronized_subset.bag')
    out_dir = args.out_dir if args.out_dir else os.path.join(base_dir, 'labels_soft')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Opening bag file (No ROS needed!): {bag_path}")
    print(f"Gaussian Target Smearing Config: sigma = {sigma:.3f} m, search cutoff = {cutoff:.3f} m (4*sigma)")
    print(f"Output directory: {out_dir}")
    print("-" * 60)

    if not os.path.exists(bag_path):
        print(f"Error: Bag file not found at {bag_path}")
        return

    typestore = get_typestore(Stores.ROS1_NOETIC)
    current_raw = None
    pair_count = 0

    with Reader(bag_path) as reader:
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            
            if topic not in ['/saved/raw_points', '/saved/rms_points']:
                continue
                
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            
            if topic == '/saved/raw_points':
                current_raw = msg
                
            elif topic == '/saved/rms_points':
                if current_raw is None:
                    continue 
                    
                pair_count += 1
                print(f"Processing Pair #{pair_count}...")

                raw_xyz = decode_pointcloud2(current_raw)
                rms_xyz = decode_pointcloud2(msg)

                # --- Gaussian Target Smearing (Thesis Chapter 5, Eq. 5.1) ---
                # Y_i = exp(- D_i^2 / (2 * sigma^2))
                target_vector = np.zeros((len(raw_xyz), 1), dtype=np.float32)

                if len(rms_xyz) > 0 and len(raw_xyz) > 0:
                    # Build tree on the RMS-selected points (positive structural anchors)
                    rms_tree = cKDTree(rms_xyz)

                    # Query Euclidean distance D_i from every raw point to nearest RMS point
                    distances, _ = rms_tree.query(raw_xyz, distance_upper_bound=cutoff, **_QUERY_KWARG)

                    # Compute continuous soft label
                    # Distances beyond cutoff return inf, where exp(-inf) = 0.0
                    soft_labels = np.exp(- (distances ** 2) / (2.0 * (sigma ** 2)))
                    target_vector = np.nan_to_num(soft_labels, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).reshape(-1, 1)

                # Save coordinates and continuous targets as dictionary
                save_dict = {
                    'pos': raw_xyz,
                    'y': target_vector
                }
                
                label_filename = os.path.join(out_dir, f"frame_{pair_count:05d}.npy")
                np.save(label_filename, save_dict)
                
                # Metrics for visibility into the continuous distribution
                high_conf = np.count_nonzero(target_vector >= 0.5)
                mid_conf = np.count_nonzero(target_vector >= 0.1)
                print(
                    f"  -> Saved {label_filename} | Raw: {len(raw_xyz)} pts, RMS anchors: {len(rms_xyz)} | "
                    f"Soft labels (Y>=0.5): {high_conf}, (Y>=0.1): {mid_conf}"
                )
                current_raw = None

    print("-" * 60)
    print("Extraction Complete! Continuous soft labels ready for PointNeXt training.")

if __name__ == '__main__':
    main()
