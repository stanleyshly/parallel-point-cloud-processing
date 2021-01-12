import cupy
import cupy as cp
from cupy import cuda
import cupyx
import numpy as np
import open3d as o3d
import time
def min_array_cupy(array):
	min_array = cp.min(array, axis=0)# get min along column
	x_min = cp.asnumpy(min_array[0]).astype(int)
	y_min = cp.asnumpy(min_array[1]).astype(int)
	z_min = cp.asnumpy(min_array[2]).astype(int)
	return x_min, y_min, z_min

def max_array_cupy(array):
	max_array = cp.max(array, axis=0)# get max along column
	x_max = cp.asnumpy(max_array[0]).astype(int)
	y_max = cp.asnumpy(max_array[1]).astype(int)
	z_max = cp.asnumpy(max_array[2]).astype(int)
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

def get_n_blocks(pt, array, current_itr, num_block_to_allocate, blocks_allocated_num ,block_x_length, block_y_length, block_z_length):
	block_array = []
	itr = 0 
	count = 0
	#for count in range(current, current+itr_num+1):
	while (current_itr+count < current_itr+num_block_to_allocate):
		#if (current+itr+1 == array.shape[0]):
		if (current_itr+itr >= array.shape[0]):
			for num in range (0, num_block_to_allocate-len(block_array)):
				block_array.append(cp.zeros(shape=(0,3)))
			return block_array, current_itr+itr, blocks_allocated_num+count, False
			

		x_min, y_min, z_min = array[current_itr+itr]
		block = find_point_in_bounding_box_cupy(pt, x_min, y_min, z_min, x_min+block_x_length, y_min+block_y_length, z_min+block_z_length)
		
		#print(x_min, y_min, z_min, x_min+block_x_length, y_min+block_y_length, z_min+block_z_length)
		if (block.shape != (0,3)):
			block_array.append(block)
			count = count + 1

		itr = itr + 1
		#print(current+count)

	return block_array, current_itr+itr, blocks_allocated_num+count, True

def allocate_n_streams(n):
	map_streams = []
	for i in range(n):
		map_streams.append(cp.cuda.stream.Stream(non_blocking=True))

	return map_streams

def largest_num_pt(array):
	highest = 0
	for current in array:
		if (current.shape[0] > highest):
			highest = current.shape[0]
	return highest
def linalg_norm (array, pt):
	return cp.linalg.norm(array-pt, axis=1)
def dist_cuda_concurrent(pt_array):
	#print(pt_array.shape[0]-1)
	dist = cp.empty(shape=(pt_array.shape[0], pt_array.shape[0]))
	
	#print(dist.shape)
	#print(dist.shape)
	for index in range (0,pt_array.shape[0]):
		dist_one = linalg_norm(pt_array, pt_array[index])
		#print(dist_one.shape, pt_array.shape[0])
		dist[index] = dist_one
	#print('end')
	#print(dist.shape)
	#print(dist)
	return dist 
def dist_itr_funct(array_block ,map_streams, memory_pool, cupy_stream):
	zs = []
	dist_array = []
	allocated_num = 0
	for current_block in array_block:
		with map_streams[allocated_num]:
			#print(current_block.shape) 
			job = dist_cuda_concurrent(current_block)
			zs.append(job)
		stop_event = map_streams[allocated_num].record()
		stop_events.append(stop_event)
		allocated_num = allocated_num + 1
	#print(len(array_block))
	#for current_stream in map_streams:
	#	with current_stream:
	#		x= 0#print(len(array_block))
	
	for i in range(len(map_streams)):
		cupy_stream.wait_event(stop_events[i])
	
	
	for current in zs:
		#print(current.shape)
		dist_array.append(current)
		
	#print(memory_pool.used_bytes()) 
	
	#print('cleaning memory')
	#del zs
	for stream in map_streams:
		memory_pool.free_all_blocks(stream=stream)
	#print(memory_pool.used_bytes()) 
	return dist_array#utput

def cp_filter_concurrent(array_block, current_index , idx, map_streams, memory_pool, cupy_stream):
	output = []
	count = 0
	for current_stream in map_streams:
		with current_stream:
			#print((type(array_block[current_index].shape)))
			#print(allocated_num)
			if (count < len(array_block)):	

				if ((type(array_block[current_index]) is tuple)):
					if (current_index < array_block[current_index].shape[0]):
						#print(idx)
						output.append(cp_filter(idx[current_index],current_index ,0.1, array_block[current_index]))
		count = count + 1
	return output
def cp_filter(idx, current_pt_index, epsilon, points):
		
	points_copy = cp.zeros(shape=points.shape)
	cp.copyto(points_copy, points)

	#print(points)


	neighbors = points[idx, :]

	#neighbors = neighbors[0, :, :]
	#print('tes')
	mean = cp.mean(neighbors, 0)
	#print(neighbors.shape)
	cov = cp.cov(neighbors.T)
	e = cp.linalg.inv(cov + epsilon * cp.eye(3))
	A = cov @ e
	b = mean - A @ mean
	points_copy[current_pt_index] = A @ points[current_pt_index] + b
	return points_copy
def cp_where_serial(array, threshold):
	idx_array = []
	for current in array:
		temp = False
		for row_idx in range (0, current.shape[0]):
			idx_this_row = cp.where(current[row_idx] < threshold)
			#print(idx_this_row)
			if temp == False:
				idx = cp.asarray(idx_this_row)
				temp = True
			if temp == True:
				idx_this_row = cp.asarray(idx_this_row)
				cp.hstack([idx, idx_this_row])
		idx_array.append(idx)
	return idx_array





start_time = time.time()

device = cupy.cuda.Device()
memory_pool = cupy.cuda.MemoryPool()
cupy.cuda.set_allocator(memory_pool.malloc)
rand = cupy.random.generator.RandomState(seed=1)
cupy_stream = cupy.cuda.stream.Stream()
pinned_mempool = cupy.get_default_pinned_memory_pool()


#x = rand.normal(size=(2048**2, 3), dtype=cupy.float32)[None]
input_pt = o3d.io.read_point_cloud("./3dLidar.txt", format='xyz')
pcd_cp = cp.asarray(input_pt.points)

x_min, y_min, z_min = min_array_cupy(pcd_cp)
x_max, y_max, z_max = max_array_cupy(pcd_cp)
#print(x_min, y_min, z_min)
#print(x_max, y_max, z_max)
block_x_length = int(round(((x_max-x_min)/8).item()))
block_y_length = int(round(((y_max-y_min)/8).item()))
block_z_length = int(round(((z_max-z_min)/8).item()))
#print(block_x_length)

block_start_array = get_array_blocks(x_min, x_max, y_min, y_max, z_min, z_max, block_x_length, block_y_length, block_z_length)
#print(block_start_array.shape)

#block_array, current_index, allocated_num, block_remain_status = get_n_blocks(pcd_cp, block_start_array, current_index, 8, allocated_num, block_x_length, block_y_length, block_z_length)

num_streams = 8
map_streams=allocate_n_streams(num_streams)

block_remain_status = True
current_index = 0
allocated_num = 0
count_allocate_num = 0
for itr in range(0, 4**3, num_streams):
	zs= []
	stop_events = []
	#create blocks
	block_array, current_index, allocated_num, block_remain_status = get_n_blocks(pcd_cp, block_start_array, current_index, num_streams, allocated_num, block_x_length, block_y_length, block_z_length)
	if block_remain_status != True:
		break
	dist_array = dist_itr_funct(block_array, map_streams, memory_pool, cupy_stream)
	
	for current in block_array:
		print(current.shape)
	for current in dist_array:
		print(current.shape)
	#print(len(block_array))
	
	
	idx_array = cp_where_serial(dist_array, 100)
	for current in idx_array:
		#for idx in current:
		print(current.shape)
	break
	#output = dist_itr_funct(block_array, highest_pt_len, count_allocate_num, map_streams, memory_pool, cupy_stream)


	#print(count_allocate_num)



device.synchronize()
elapsed_time = time.time() - start_time
print('elapsed time', elapsed_time)
#print('total bytes', memory_pool.total_bytes())