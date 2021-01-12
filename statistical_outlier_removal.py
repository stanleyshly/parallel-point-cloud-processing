import cupy as cp
import numpy as np
import open3d as o3d
from tqdm import tqdm
import time


pcd_raw = o3d.io.read_point_cloud("./3dLidar.txt", format='xyz')
pcd_raw = pcd_raw.uniform_down_sample(every_k_points=1)
pcd_raw_np =cp.asarray(pcd_raw.points)
#print(pcd_raw_np)



def outlier_remove(pcd_raw_np, k_neighbors, std_alpha):
	deleted_indices = cp.empty(shape=[1])
	clean_pt = cp.empty(shape=[0,3])
	#print(pcd_raw_np.shape)
	pbar = tqdm(total=len(pcd_raw_np), smoothing=0.0)
	for current in  (range(len(pcd_raw_np))):
		#print(pcd_raw_np[current])
		#x = 1+1
		# get distance of all points
		dist = cp.linalg.norm(pcd_raw_np-pcd_raw_np[current], axis=1)
		# get k smallest
		idx_k_smallest = cp.argpartition(dist, k_neighbors)
		k_smallest = pcd_raw_np[idx_k_smallest[:k_neighbors]]
		#find average distance
		mean_dist = cp.mean(k_smallest)
		#find standard deviation of distance
		std_dist = cp.std(k_smallest)
		#print(average)
		#threshold
		t = mean_dist + std_alpha * std_dist
		#print(t)
		#get all indices that are greater than a certain value
		deleted = cp.where( dist[idx_k_smallest[:k_neighbors]] >t)[0] #cp.delete(dist[idx_k_smallest[:k_neighbors]], dist > t)
		#print(deleted)
		if (deleted.size != 0):
			#print(deleted)
			deleted_indices = cp.concatenate([deleted_indices, deleted])
		#print(deleted_indices)
		pbar.update(1)

	#remove all duplicates
	clean_deleted_indices =cp.unique(deleted_indices)
	
	#delete by index
	for current in range(len(pcd_raw_np)):
		if current  not in clean_deleted_indices:
			clean_pt = cp.concatenate([clean_pt, pcd_raw_np[current].reshape(-1,3)])
	#print(clean_pt.shape)
	pbar.close()
	return clean_pt



start_time = time.time()

clean_pt = outlier_remove(pcd_raw_np, 20, -100)
np.savetxt("./clean.txt", cp.asnumpy(clean_pt))


print("--- %s seconds ---" % (time.time() - start_time))
