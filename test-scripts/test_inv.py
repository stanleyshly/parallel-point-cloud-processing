import cupy
import cupyx
import cupy as cp
from cupy import cuda

import numpy
import time
def _as_batch_mat(x):
	return x.reshape(len(x), x.shape[1], -1)


def _mat_ptrs(a):
	if len(a) == 1:
		return cupy.full((1,), a.data.ptr, dtype=numpy.uintp)
	else:
		stride = a.strides[0]
		ptr = a.data.ptr
		out = cupy.arange(ptr, ptr + stride * len(a), stride, dtype=numpy.uintp)
		return out


def _get_ld(a):
	strides = a.strides[-2:]
	trans = numpy.argmin(strides)
	return trans, int(max(a.shape[trans - 2], max(strides) // a.itemsize))


def inv_gpu(b):
	# We do a batched LU decomposition on the GPU to compute the inverse
	# Change the shape of the array to be size=1 minibatch if necessary
	# Also copy the matrix as the elments will be modified in-place
	a = _as_batch_mat(b).copy()
	n = a.shape[1]
	n_matrices = len(a)
	# Pivot array
	p = cupy.empty((n, n_matrices), dtype=numpy.int32)
	# Output array
	c = cupy.empty_like(a)
	# These arrays hold information on the execution success
	# or if the matrix was singular
	info = cupy.empty(n_matrices, dtype=numpy.int32)
	ap = _mat_ptrs(a)
	cp = _mat_ptrs(c)
	_, lda = _get_ld(a)
	_, ldc = _get_ld(c)
	handle = cuda.Device().cublas_handle
	cuda.cublas.sgetrfBatched(
		handle, n, ap.data.ptr, lda, p.data.ptr, info.data.ptr, n_matrices)
	cuda.cublas.sgetriBatched(
		handle, n, ap.data.ptr, lda, p.data.ptr, cp.data.ptr, ldc,
		info.data.ptr, n_matrices)
	return c, info

device = cupy.cuda.Device()
memory_pool = cupy.cuda.MemoryPool()
cupy.cuda.set_allocator(memory_pool.malloc)
rand = cupy.random.generator.RandomState(seed=1)

n = 4
zs = []
map_streams = []
stop_events = []
reduce_stream = cupy.cuda.stream.Stream()
for i in range(n):
	map_streams.append(cupy.cuda.stream.Stream())

start_time = time.time()
#def linalg_norm (array, pt):
#    return cp.linalg.norm(array-pt, axis=1)

def dist_cuda_concurrent(pt_array):
	#print(pt_array.shape[0]-1)
	dist = cp.empty(shape=[pt_array.shape[0], pt_array.shape[0]])
	
	print(pt_array.shape[0])
	#print(dist.shape)
	for index in range (0,pt_array.shape[0]):
		dist = cp.linalg.norm(pt_array- pt_array[index], axis = 1)
	#print(dist_one.shape, pt_array.shape[0])
	#dist[index] = dist_one
	#print('end')
	#print(dist.shape)
	#print(dist)
	return dist 
#cupyx.seterr(linalg='ignore') # disable cublas check, which waits, creating series
for stream in map_streams:
	with stream:
		'''
		x = cupy.array([
	[1, 0, 1], 
	[0, 1, 0], 
	[0, 0, 1]]).astype(cupy.float32)[None]''' #
		x = rand.normal(size=(64**2, 3), dtype=cupy.float32)
		#print(x)
		#y = rand.normal(size=(1024**2, 1))

		z = dist_cuda_concurrent(x)
		#z = numpy.linalg.inv(x)
		
		zs.append(z)
	stop_event = stream.record()
	stop_events.append(stop_event)

# Block the `reduce_stream` until all events occur. This does not block host.
# This is not required when reduction is performed in the default (Stream.null)
# stream unless streams are created with `non_blocking=True` flag.
for i in range(n):
	reduce_stream.wait_event(stop_events[i])

# Reduce
with reduce_stream:
	z = zs
	print(z)

device.synchronize()
elapsed_time = time.time() - start_time
print('elapsed time', elapsed_time)
print('total bytes', memory_pool.total_bytes())

# Free all blocks in the memory pool of streams
for stream in map_streams:
	memory_pool.free_all_blocks(stream=stream)
print('total bytes', memory_pool.total_bytes())