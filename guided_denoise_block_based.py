import numpy as cp
import numpy as np
import open3d as o3d
import pandas as pd
import time
from tqdm import tqdm
import warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')

def main():
	#x = rand.normal(size=(2048**2, 3), dtype=cupy.float32)[None]
	input_pt = o3d.io.read_point_cloud("./clean.xyz")#, format='xyz')
	input_pt = add_noise(input_pt, 1)
	#o3d.visualization.draw_geometries([input_pt])

	input_pt = pd.DataFrame(input_pt.points)
	df_elements = input_pt.sample(input_pt.shape[0])#n=10000)


	pcd_cp = (df_elements.to_numpy())#(input_pt.points)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pcd_cp)
	#o3d.visualization.draw_geometries([pcd])
	start_time = time.time()
	returned_pt = filter_block_based(pcd_cp, blocknumx=3, blocknumy=3, blocknumz=3, radius=15, epsilon=250)
	returned_pt = filter_block_based(returned_pt, blocknumx=4, blocknumy=4, blocknumz=4, radisu=15, epsilon=250)

	device.synchronize()
	elapsed_time = time.time() - start_time
	print('elapsed time', elapsed_time)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(creturned_pt)
	o3d.visualization.draw_geometries([pcd])
	o3d.io.write_point_cloud("test_pd.ply", pcd)
#print('total bytes', memory_pool.total_bytes())
def min_array_cupy(array):
	min_array = cp.min(array, axis=0)# get min along column
	x_min = (min_array[0]).astype(int)
	y_min = (min_array[1]).astype(int)
	z_min = (min_array[2]).astype(int)
	return x_min, y_min, z_min

def max_array_cupy(array):
	max_array = cp.max(array, axis=0)# get max along column
	x_max = (max_array[0]).astype(int)
	y_max = (max_array[1]).astype(int)
	z_max = (max_array[2]).astype(int)
	return x_max, y_max, z_max
def get_array_blocks(x_min, x_max, y_min, y_max, z_min, z_max,block_x_length, block_y_length, block_z_length):
		
		count = 0 
		for x_current_block_length in range (x_min, x_max, block_x_length):
			for y_current_block_length in range (y_min, y_max, block_y_length):
				for z_current_block_length in range (z_min, z_max, block_z_length):
						count = count + 1
						#print(x_current_min,  x_current_max, y_current_min, y_current_max, z_current_min, z_current_max)
		block_array = cp.zeros(shape=(count+1,3))
		#print(block_array.shape)
		count = 0 
		for x_current_block_length in range (x_min, x_max, block_x_length):
			for y_current_block_length in range (y_min, y_max, block_y_length):
				for z_current_block_length in range (z_min, z_max, block_z_length):
						count = count + 1
						block_array[count][0] = x_current_block_length
						block_array[count][1] = y_current_block_length
						block_array[count][2] = z_current_block_length

						
		return block_array
def find_point_in_bounding_box_cupy(input_array, x_current_min, y_current_min, z_current_min, x_current_max, y_current_max, z_current_max):

	ll = cp.array([x_current_min, y_current_min, z_current_min])  # lower-left
	ur = cp.array([x_current_max, y_current_max, z_current_max])  # upper-right

	inidx = cp.all(cp.logical_and(ll <= input_array, input_array <= ur), axis=1)
	inbox = input_array[inidx]
	outbox = input_array[cp.logical_not(inidx)]
	return inbox

def find_pt_around(input_array, x_current_min, y_current_min, z_current_min, x_current_max, y_current_max, z_current_max, radius):
	#pt = cp.zeros(shape=(0,3))
	#count = 0 
	inner = find_point_in_bounding_box_cupy(input_array, x_current_min, y_current_min, z_current_min, x_current_max, y_current_max, z_current_max)
	outer = find_point_in_bounding_box_cupy(input_array, x_current_min-radius, y_current_min-radius, z_current_min-radius, x_current_max+radius, y_current_max+radius, z_current_max+radius)
	#print('innner '+ str(inner.shape))
	#print('outer' + str(outer.shape))
	count = 0
	indexArr = []
	for current_pt in outer:
		if ((current_pt == inner).all(1).any()):
			indexArr.append(count)
		count = count  + 1
	indexArr = np.array(indexArr).astype(int)
	#print('count ' + str(count))
	#print('shape index array ' + str((indexArr).shape))
	outer = outer
	#print(indexArr)
	outer=np.delete(outer, indexArr, axis=0)
	outer = cp.asarray(outer)
	return inner, outer
	#return count
def np_filter(idx_inner, idx_outer, epsilon, points, points_copy, points_outer, pointer_outer_copy, i):
	neighbors_inner = points[idx_inner, :]
	neighbors_inner = neighbors_inner[0, :, :]
	
	neighbors_outer = points_outer[idx_outer, :]
	neighbors_outer = neighbors_outer[0, :, :]

	#print(neighbors_inner.shape[0]/(neighbors_inner.shape[0]+neighbors_outer.shape[0]))
	#print(neighbors_outer.shape[0]/(neighbors_inner.shape[0]+neighbors_outer.shape[0]))
	#print(neighbors_outer.shape, neighbors_inner.shape, cp.concatenate((neighbors_outer, neighbors_inner)).shape)
	#mean_inner = cp.average(neighbors_inner, 0)#,  weights=[neighbors_outer.shape[0]/(neighbors_inner.shape[0]+neighbors_outer.shape[0])])
	#mean_outer = cp.average(neighbors_outer, 0)#, weights=[neighbors_inner.shape[0]/(neighbors_inner.shape[0]+neighbors_outer.shape[0])])#])
	neighbors = cp.concatenate((neighbors_outer, neighbors_inner))
	#print(neighbors.shape)
	mean = cp.mean(neighbors, 0)#( cp.array([ mean_inner, mean_outer ]), axis=0)# ,weights=[neighbors_outer.shape[0]/(neighbors_inner.shape[0]+neighbors_outer.shape[0]), neighbors_inner.shape[0]/(neighbors_inner.shape[0]+neighbors_outer.shape[0])])

	
	#print('shape neighbrs', str(neighbors_outer.shape), str(neighbors_inner.shape))
	#
	#print(neighbors.shape)
	cov = cp.cov(neighbors.T)
	e = cp.linalg.inv(cov + epsilon * cp.eye(3))

	A = cov @ e
	b = mean - A @ mean
	points_copy[i] = A @ points[i] + b
	return points_copy


def guided_filter(points_inner_copy, points_inner, points_outer_copy, points_outer, num_points_inner, num_points_outer, radius ,epsilon, pbar):
	#pcd = o3d.geometry.PointCloud()

	for i in range(num_points_inner):
		
			#print(points_inner.shape)
		dist_inner = cp.linalg.norm(points_inner_copy-points_inner[i], axis=1)
		dist_outer = cp.linalg.norm(points_outer_copy-points_inner[i], axis=1)
			#print(dist)
		idx_inner = cp.where(dist_inner < radius)
		idx_outer = cp.where(dist_outer < radius)
		points_inner_copy = cp.array(inner)
		points_inner = inner
		points_outer_copy = cp.array(outer)
		points_outer = outer
		num_points_inner = points_inner.shape[0]
		num_points_outer = points_outer.shape[0]
		
		#print('idx', str((idx_inner[0])), str((idx_outer[0])))
		try:	#print(idx_inner)
		
			idx_inner = cp.asarray(idx_inner)
			idx_outer = cp.asarray(idx_outer)
			#print(idx.shape)
			#print('enter')
			points_inner_copy = np_filter(idx_inner, idx_outer, epsilon, points_inner, points_inner_copy, points_outer, points_outer_copy, i)
		except:
			pass

		pbar.update(1)

			

	
	#pcd.points = o3d.utility.Vector3dVector(cp.asnumpy(points_inner_copy))
	return points_inner_copy 
def add_noise(pcd, sigma):
    points = cp.asarray(pcd.points)
    noise = sigma * cp.random.randn(points.shape[0], points.shape[1])
    points += noise

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd





def filter_block_based(pcd_cp,blocknumx=1, blocknumy=1, blocknumz=1, radius=0, epsilon=100):
	

	x_min, y_min, z_min = min_array_cupy(pcd_cp)
	x_max, y_max, z_max = max_array_cupy(pcd_cp)
	#print(x_min, y_min, z_min)
	#print(x_max, y_max, z_max)
	block_x_length = int(round(((x_max-x_min)/blocknumx).item()))
	block_y_length = int(round(((y_max-y_min)/blocknumy).item()))
	block_z_length = int(round(((z_max-z_min)/blocknumz).item()))
	#print(block_x_length)

	block_start_array = get_array_blocks(x_min, x_max, y_min, y_max, z_min, z_max, block_x_length, block_y_length, block_z_length)
	clean_pt = cp.empty(shape=[0,3])

	print('Denoising with Block Based Method.... Processing:',str(block_start_array.shape[0]),'blocks')
	pbar = tqdm(total=pcd_cp.shape[0], smoothing=0.0)# the 0.0 smoothing sets to average speed

	for itr in (range(block_start_array.shape[0])):
		current =  block_start_array[itr]
		if (current.shape[0] != 0):
			print(current)
			inner, outer =find_pt_around(pcd_cp ,current[0], current[1], current[2], current[0]+ block_x_length, current[1]+block_y_length, current[2]+block_z_length, radius)
			#print(inner.shape, outer.shape)
			if (inner.shape[0] != 0):
				'''
				# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
				pcd = o3d.geometry.PointCloud()
				pcd.points = o3d.utility.Vector3dVector(cp.asnumpy(inner))
				o3d.visualization.draw_geometries([pcd])

				pcd = o3d.geometry.PointCloud()
				pcd.points = o3d.utility.Vector3dVector(cp.asnumpy(outer))
				o3d.visualization.draw_geometries([pcd])
				'''

				pcd = guided_filter(points_inner_copy, points_inner, points_outer_copy, points_outer, num_points_inner, num_points_outer, radius , epsilon, pbar)
				clean_pt = cp.concatenate((clean_pt, pcd))
	pbar.close()
	return clean_pt


