import numpy as np

import open3d as o3d
from tqdm import tqdm
import time


def filter_single_process(pcd, radius=0, epsilon=0):
    points_copy = np.array(pcd)
    points = np.asarray(pcd)
    num_points = pcd.shape[0]
    # filtering multiple times will reduce the noise significantly
    # but may cause the points distribute unevenly on the surface.
    #o3d.visualization.draw_geometries([pcd])
    pcd = guided_filter(pcd, points_copy, points, num_points, radius ,epsilon)
    return pcd
def np_filter(idx, epsilon, points, points_copy, i):
    neighbors = points[idx, :]
    neighbors = neighbors[0, :, :]
    mean = np.mean(neighbors, 0)
    #print(neighbors.shape)
    cov = np.cov(neighbors.T)
    e = np.linalg.inv(cov + epsilon * np.eye(3))

    A = cov @ e
    b = mean - A @ mean
    points_copy[i] = A @ points[i] + b
    return points_copy


def guided_filter(pcd, points_copy, points, num_points, radius, epsilon):
    pbar = tqdm(total=num_points, smoothing=0.0)
    for i in (range(num_points)):
        #print(pcd.points[i])

        #k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        dist = np.linalg.norm((points-pcd[i]), axis=1)
        dist = np.abs(dist)
        #print(dist)
        idx = np.where(dist < radius)
       
        try:
            idx = np.asarray(idx)
            #print(idx.shape)
            points_copy = np_filter(idx, epsilon, points, points_copy, i)
        except:
            pass
        pbar.update(1)
        #print(type(idx))
        #if k < 3:
        #    continue
            

    pbar.close()
#pcd.points = o3d.utility.Vector3dVector(points_copy)
    return pcd 


def add_noise(pcd, sigma):
    points = np.asarray(pcd.points)
    noise = sigma * np.random.randn(points.shape[0], points.shape[1])
    points += noise

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asnumpy(points))

    return pcd

