import numpy as cp
import numpy as np
import open3d as o3d
import pandas as pd
import time
from tqdm import tqdm
import multiprocessing 
import warnings
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')


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
		block_array = cp.zeros(shape=(count,3))
		#print(block_array.shape)
		count = 0 
		for x_current_block_length in range (x_min, x_max, block_x_length):
			for y_current_block_length in range (y_min, y_max, block_y_length):
				for z_current_block_length in range (z_min, z_max, block_z_length):
						
						block_array[count][0] = x_current_block_length
						block_array[count][1] = y_current_block_length
						block_array[count][2] = z_current_block_length
						count = count + 1

						
		return block_array
def find_point_in_bounding_box_cupy(input_array, x_current_min, y_current_min, z_current_min, x_current_max, y_current_max, z_current_max):

	ll = cp.array([x_current_min, y_current_min, z_current_min])  # lower-left
	ur = cp.array([x_current_max, y_current_max, z_current_max])  # upper-right

	inidx = cp.all(cp.logical_and(ll <= input_array, input_array <= ur), axis=1)
	inbox = input_array[inidx]
	#outbox = input_array[cp.logical_not(inidx)]
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

	outer=np.delete(outer, indexArr, axis=0)
	return inner, outer
	#return count
def np_filter(idx_inner, idx_outer, epsilon, points, points_copy, points_outer, pointer_outer_copy, i):
	neighbors_inner = points[idx_inner, :]
	neighbors_inner = neighbors_inner[0, :, :]
	
	neighbors_outer = points_outer[idx_outer, :]
	neighbors_outer = neighbors_outer[0, :, :]

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


def guided_filter(points_inner_copy, points_inner, points_outer_copy, points_outer, num_points_inner, num_points_outer, radius ,epsilon):
	#pcd = o3d.geometry.PointCloud()
	for i in range(num_points_inner):
		
			#print(points_inner.shape)
		dist_inner = cp.linalg.norm(points_inner_copy-points_inner[i], axis=1)
		dist_outer = cp.linalg.norm(points_outer_copy-points_inner[i], axis=1)
			#print(dist)
		idx_inner = cp.where(dist_inner < radius)
		idx_outer = cp.where(dist_outer < radius)
		#print('idx', str((idx_inner[0])), str((idx_outer[0])))
		try:	#print(idx_inner)
		
			idx_inner = cp.asarray(idx_inner)
			idx_outer = cp.asarray(idx_outer)
			#print(idx.shape)
			#print('enter')
			points_inner_copy = np_filter(idx_inner, idx_outer, epsilon, points_inner, points_inner_copy, points_outer, points_outer_copy, i)
			return points_inner_copy
		except:
			pass


			

	
	#pcd.points = o3d.utility.Vector3dVector(cp.asnumpy(points_inner_copy))
	return points_inner_copy 
def add_noise(pcd, sigma):
    points = cp.asarray(pcd.points)
    noise = sigma * cp.random.randn(points.shape[0], points.shape[1])
    points += noise

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd

def multi_run_wrapper(args):
   return thread_mulitprocessing(*args)

def thread_mulitprocessing(pcd_cp, current, radius, epsilon, block_x_length, block_y_length,block_z_length):
	if (current.shape[0] != 0):
		#print(current)
		inner, outer =find_pt_around(pcd_cp ,current[0], current[1], current[2], current[0]+ block_x_length, current[1]+block_y_length, current[2]+block_z_length, radius)
		#print(inner.shape, outer.shape)
		#return inner
		
		if (inner.shape[0] != 0):
			points_inner_copy = cp.array(inner)
			points_inner = inner
			points_outer_copy = cp.array(outer)
			points_outer = outer
			num_points_inner = points_inner.shape[0]
			num_points_outer = points_outer.shape[0]
			pcd = guided_filter(points_inner_copy, points_inner, points_outer_copy, points_outer, num_points_inner, num_points_outer, radius , epsilon)	
			return pcd
		else:
			return None
	else:
		return None
	

def multiprocessing_filter_block_based(pcd_cp,blocknumx=1, blocknumy=1, blocknumz=1, radius=0, epsilon=100, cores=1):
	

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

	print('Denoising with Mulitprocessing Block Based Method:','cores', str(cores),",", str(block_start_array.shape[0]),'blocks')
	pbar = tqdm(total=pcd_cp.shape[0], smoothing=0.0)# the 0.0 smoothing sets to average speed
	count = 1
	array = []
	num = 0 
	for itr in (range(block_start_array.shape[0])):
		current =  block_start_array[itr]

		array.append((pcd_cp, current, radius, epsilon, block_x_length, block_y_length,block_z_length))
		#

		if (count %cores == 0):
			#pcd=thread_mulitprocessing(pcd_cp, current, radius, epsilon, block_x_length, block_y_length,block_z_length)
			with multiprocessing.Pool(processes=cores) as pool:
				#print(array[0].shape)
				results = pool.map(multi_run_wrapper, array)
			#print(results)
			for processed in results:

				if processed is not None:
					clean_pt= np.concatenate((processed, clean_pt))
					num = num + processed.shape[0]

			count = 0
			array = []
		count = count + 1
	#pbar.close()
	#print(array)
	if (len(array) != 0):
		with multiprocessing.Pool(processes=len(array)) as pool:
			#print(len(array))
			results = pool.map(multi_run_wrapper, array)
			#print(results)
			for processed in results:

				if processed is not None:
					#print(processed.shape, clean_pt.shape)
					try:
						clean_pt = np.concatenate((processed, clean_pt))
						num = num + processed.shape[0]
						
					except:
						pass
	clean_pt = clean_pt[np.logical_not(np.isnan(clean_pt))].reshape(-1,3)
	#print(num)
	pbar.update(pcd_cp.shape[0])
	pbar.close()
	return clean_pt


