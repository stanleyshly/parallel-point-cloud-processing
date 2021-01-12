import numpy as np
import numpy as cp
import open3d as o3d
from tqdm import tqdm
import time



def block_radial_remove(pcd_cp, blocknumx=1, blocknumy=1, blocknumz=1, radius=1, nb_neighbors=1):
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

    print('Removing Outliers with Block Based Method... Processing:',str(block_start_array.shape[0]), "blocks")
    pbar = tqdm(total=pcd_cp.shape[0], smoothing=0.0)# the 0.0 smoothing sets to average speed
    
    for itr in (range(block_start_array.shape[0])):
        current =  block_start_array[itr]
        if (current.shape[0] != 0):
            
            inner, outer =find_pt_around(pcd_cp ,current[0], current[1], current[2], current[0]+ block_x_length, current[1]+block_y_length, current[2]+block_z_length, radius)
            #print(inner.shape, outer.shape)
            if (inner.shape[0] != 0):
                num_points = inner.shape[0]
                #print(num_points)
                pcd = radial_remove(inner, outer, num_points, pbar, nb_neighbors=nb_neighbors, radius=radius)
                clean_pt = cp.concatenate((clean_pt, pcd))
    pbar.close()
    return clean_pt

def radial_remove(pcd, outer_pcd, num_points, pbar, nb_neighbors=0, radius=5):
    ''' nb_neighbors specifies minimum number of points'''
    
    #deleted_indices = cp.empty(shape=[1])
    idx_arr = []
    clean_pt = cp.empty(shape=[0,3])

    inner_outer_pt = cp.concatenate((pcd,outer_pcd))
    for i in (range(num_points)):
        #print(pcd.points[i])

        #k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
        dist = cp.linalg.norm(inner_outer_pt-cp.asarray(pcd[i]), axis=1)
        dist = cp.abs(dist)
        #print(dist)
        idx = cp.where(dist < radius)
        
        #remove if less than nb_neighbors
        if len(idx[0])<nb_neighbors or (cp.asarray(pcd[i])[0] == 0 and cp.asarray(pcd[i])[1] == 0 and cp.asarray(pcd[i])[2] == 0):
            #deleted_indices = cp.concatenate([deleted_indices, cp.asarray([i])])
            idx_arr.append(i)
        pbar.update(1)
    #print('gdsfgsfgsdfg', num_points)
    #print(idx_arr)    #clean_pt = np.delete(pcd, cp.asnumpy(deleted_indices).astype(int).tolist(), axis=0)
    clean_pt = np.delete(pcd, idx_arr, axis=0)
    clean_pt = cp.asarray(clean_pt)
        
    return clean_pt



    

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

    #print(indexArr)
    outer=np.delete(outer, indexArr, axis=0)
    outer = cp.asarray(outer)
    return inner, outer
    #return count
