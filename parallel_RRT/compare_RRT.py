import math
import numpy as np
from numba import cuda
from tqdm import tqdm
import matplotlib.pyplot as plt

#import parallel_RRT_test


N= 32
TPB = 256

@cuda.jit(device = True )
def num_iteration(num):
	new = np.ones(num)
	return new

@cuda.jit
def kernel_para_RRT(d_diff, d_f):
	i= cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
	if i < d_diff.shape[0]:
	#d_diff[0] = 1
		d_diff[i] = parallel_RRT_test.main()

def wrap_fun_para_comp_RRT(f):
	n = f
	d_f = cuda.to_device(n)
	d_diff = cuda.device_array_like(d_f)
	blocks, threads = (n+TPB-1)//TPB, TPB
	kernel_para_RRT[blocks, threads](d_diff, d_f)
	return d_diff.copy_to_host()

def main():
	num =2
	res = wrap_fun_para_comp_RRT(num)
	print(res)

#parallel_RRT_test.main()
if __name__ == '__main__':
	main()