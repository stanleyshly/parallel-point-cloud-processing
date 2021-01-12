from guided_denoise import filter_single_process
from guided_denoise_block_based import filter_block_based
from guided_denoise_block_based_multiprocessing import multiprocessing_filter_block_based
from radius_outlier_removal import radial_remove
from radius_outlier_removal_block_based import block_radial_remove
from radius_outlier_removal_block_based_multiprocessing import block_radial_remove_multiprocessing
import os
import open3d as o3d
import numpy as np
import numpy as cp
if __name__ == "__main__":
	for root, dirs, files in os.walk("./raw_pt/"):
		for file in files:
			if file.endswith(".xyz"):

				pt_dir = os.path.join(root, file)
				print(pt_dir)
				pcd = o3d.io.read_point_cloud(pt_dir, format='xyz')
				pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, 20)
				pcd_cp =np.asarray(pcd.points)
				#clean_pt = filter_single_process(pcd_cp, radius=0.15, epsilon=250)
				#print(clean_pt.shape)
				#clean_pt = multiprocessing_filter_block_based(pcd_cp, blocknumx=2, blocknumy=2, blocknumz=2, radius=0.15, epsilon=250, cores=1)
				#print(clean_pt.shape)
				#clean_pt = multiprocessing_filter_block_based(pcd_cp, blocknumx=2, blocknumy=2, blocknumz=2, radius=0.15, epsilon=250, cores=8)
				#print(clean_pt.shape)


				#clean_pt = radial_remove(pcd_cp, nb_neighbors=4, radius=5)
				clean_pt = block_radial_remove_multiprocessing(pcd_cp, blocknumx=4, blocknumy=4, blocknumz=4, nb_neighbors=1, radius=1, cores=8)
				print(clean_pt.shape)
				np.savetxt("./clean_pt/"+file, clean_pt)
				#pcd = o3d.geometry.PointCloud()
				#pcd.points = o3d.utility.Vector3dVector(cp.asnumpy(clean_pt))
				#o3d.visualization.draw_geometries([pcd])
				#o3d.io.write_point_cloud("./clean_pt/"+file, pcd)


				
