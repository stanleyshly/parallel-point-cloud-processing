import numpy as np
import numpy as cp
import open3d as o3d
from tqdm import tqdm
import time



def radial_remove(pcd, nb_neighbors=0, radius=5):
    ''' nb_neighbors specifies minimum number of points'''
    num_points = pcd.shape[0]
    pbar = tqdm(total=num_points, smoothing=0.0)
    deleted_indices = cp.empty(shape=[1])
    clean_pt = cp.empty(shape=[0,3])
    for i in (range(num_points)):
        #print(pcd.points[i])

        #k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        dist = cp.linalg.norm(pcd-cp.asarray(pcd[i]), axis=1)
        dist = cp.abs(dist)
        #print(dist)
        idx = cp.where(dist < radius)
        
        #remove if less than nb_neighbors
        if len(idx[0])<nb_neighbors or (cp.asarray(pcd[i])[0] == 0 and cp.asarray(pcd[i])[1] == 0 and cp.asarray(pcd[i])[2] == 0):
            deleted_indices = cp.concatenate([deleted_indices, cp.asarray([i])])
        pbar.update(1)


    clean_pt = np.delete(pcd, deleted_indices.astype(int).tolist(), axis=0)
    clean_pt = cp.asarray(clean_pt)
    #print(clean_pt.shape)
    pbar.close()
    return clean_pt 
