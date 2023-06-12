import os, sys, argparse, csv, copy
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm
import pandas as pd

def ICP(source, target, threshold, init_pose, iteration=30):
    # implement iterative closet point and return transformation matrix
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration)
    )
    #print(reg_p2p)
    #print(reg_p2p.transformation)
    return reg_p2p.transformation

def csv_reader(filename):
    # read csv file into numpy array
    data = np.loadtxt(filename, delimiter=',')
    return data

def numpy2pcd(arr):
    # turn numpy array into open3d point cloud format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


if __name__ == '__main__':

    with open(os.path.join('./private/ITRI_DLC2/test2', 'localization_timestamp.txt'), 'r') as f:
        timestamps = f.readlines()

    timestamps = [timestamp.strip() for timestamp in timestamps]

    Dxy= []
    for timestamp in tqdm(timestamps):
        path_name = './private/test2/dataset/' + timestamp
        ini_path_name = '.private/ITRI_DLC2/test2/new_init_pose'+ timestamp
        # Target point cloud
        target = csv_reader(f'{path_name}/sub_map.csv')
        target_pcd = numpy2pcd(target)
        # Initial pose
        init_pose = csv_reader(f'{path_name}/initial_pose.csv')
        # Source point cloud
        #TODO: Read your point cloud here#
        try:
            source = csv_reader(f'{path_name}/output_XYZ_final.csv')
            source_pcd = numpy2pcd(source)
        
            
            # Implement ICP
            transformation = ICP(source_pcd, target_pcd, threshold=0.02, init_pose=init_pose)
            pred_x = transformation[0,3]
            pred_y = transformation[1,3]
            Dxy.append([pred_x, pred_y])
        except:
            Dxy.append([pred_x, pred_y])
        #print(pred_x, pred_y)
        my_df = pd.DataFrame(Dxy) 
        
        my_df.to_csv('./private/solution/test2/pred_pose.txt', index=False, header=False, sep='\t')