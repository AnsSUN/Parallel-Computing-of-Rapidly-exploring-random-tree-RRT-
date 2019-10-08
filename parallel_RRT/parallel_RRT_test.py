import math
import numpy as np
from numba import cuda
from tqdm import tqdm
import matplotlib.pyplot as plt


TPB = 256 # number of threads in a block

# device function_NN
@cuda.jit(device=True)
def euc_distance_2d_device(x1,y1,x2,y2):
    d = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return d

@cuda.jit()
def distanceKernel(d_out,x,y,d_V,nov):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    if i < nov:
        d_out[i] = euc_distance_2d_device(x,y,d_V[0,i],d_V[1,i])

# wrapper function_NN
def dArray(x,y,V,nov):
    d_V = cuda.to_device(V) # copies the input data to a device array on the GPU
    d_distance = cuda.device_array(nov) # creates an empty array to hold the output
    BPG = (nov + TPB - 1)//TPB # computes number of blocks
    distanceKernel[BPG,TPB](d_distance,x,y,d_V,nov)
    return d_distance.copy_to_host()

# kernel for CC
@cuda.jit()
def ccKernel(d_out,x,y,d_O,all_radii):
    i = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    #flag = 1
    if i < all_radii.size:# and flag ==1:
        d_out[i] = euc_distance_2d_device(x,y,d_O[0,i],d_O[1,i])>all_radii[i] # should be 1 for no collision
        # flag = d_out[i]

# wrapper function_CC
def dArray_CC(x,y,obs_coors,allowable_radii):
    noo = allowable_radii.size
    d_all_radii = cuda.to_device(allowable_radii) # copies the input data to a device array on the GPU
    d_O = cuda.to_device(obs_coors) # copies the input data to a device array on the GPU
    d_collision = cuda.device_array(noo) # creates an empty array to hold the output
    BPG = (noo + TPB - 1)//TPB # computes number of blocks
    ccKernel[BPG,TPB](d_collision,x,y,d_O,d_all_radii)
    return d_collision.copy_to_host()

def euc_distance_2d(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2+(y2-y1)**2)

def nearest_neighbor_2d_parallel(x,y,V,nov):
    distance = dArray(x,y,V,nov)
    ind_min = np.argmin(distance)
    min_dis = distance[ind_min]
    return [min_dis,ind_min]

def collision_check_parallel(x,y,obstacle_coordinates,obstacle_radii):
    allowable_radii = obstacle_radii*2/np.sqrt(3)
    flag = 0 # means no collision
    if all(dArray_CC(x,y,obstacle_coordinates,allowable_radii)):
        flag = 1
    return flag

def draw_circle(xc,yc,r):
    t = np.arange(0,2*np.pi,.05)
    x = xc+r*np.sin(t)
    y = yc+r*np.cos(t)
    plt.plot(x,y,c='blue')

#@cuda.jit(device=True)
def main(num_tree):
    max_iter = 600
    epsilon = 4 # step size

    flag = 0 # for finding a connectivity path

    # initial and goal points/states
    x0 = 10
    y0 = 10
    x_goal = 90
    y_goal = 90
    plt.figure(figsize=[10,10])
    plt.scatter([x0,x_goal],[y0,y_goal],c='r',marker="P")

    # obstacle info
    noo = 16 # no. of obstacles
    radius = np.sqrt(3)/2*epsilon
    obs_radii = radius*np.ones(noo)
    obs_coors = 100*np.random.rand(2,noo) # position of obstacles
    for i in range(0,noo):
        draw_circle(obs_coors[0,i],obs_coors[1,i],obs_radii[i])

    if euc_distance_2d(x0,y0,x_goal,y_goal)<epsilon:
        flag = 1
        plt.plot([x0,x_goal],[y0,y_goal],c='black')
    else:
        vertices = np.zeros([2,max_iter+1])
        A = -np.ones([max_iter+1,max_iter+1])
        vertices[0,0] = x0
        vertices[1,0] = y0
        A[0,0] = 0

    nov = 0 # no. of vertices except the initial one
    i = 0
    while flag==0 and i<max_iter:
        i += 1
        x_rand= 100*np.random.rand(1)
        y_rand= 100*np.random.rand(1)
        xy_rand = np.array([x_rand,y_rand]).reshape(2,)
        [min_dis,p_near] = nearest_neighbor_2d_parallel(x_rand[0],y_rand[0],vertices,nov+1)
        if min_dis<epsilon:
            x_new = x_rand
            y_new = y_rand
        else: # interpolate
            r = epsilon/min_dis # ratio
            x_new = vertices[0,p_near]+r*(x_rand-vertices[0,p_near])
            y_new = vertices[1,p_near]+r*(y_rand-vertices[1,p_near])
        if collision_check_parallel(x_new[0],y_new[0],obs_coors,obs_radii):
            nov = nov+1
            vertices[0,nov] = x_new
            vertices[1,nov] = y_new
            plt.scatter(x_new,y_new,c='g')
            plt.plot([vertices[0,p_near],x_new],[vertices[1,p_near],y_new],c='black')
            A[nov,:] = A[p_near,:]
            A[nov,nov] = nov
            if euc_distance_2d(x_new,y_new,x_goal,y_goal)<epsilon:
                nov = nov+1
                A[nov,:] = A[nov-1,:]
                A[nov,nov] = nov
                vertices[0,nov] = x_goal
                vertices[1,nov] = y_goal
                plt.plot([x_new,x_goal],[y_new,y_goal],c='black')
                flag = 1
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.axis('scaled')

    if flag ==1 and nov!=0:
        B = np.zeros(nov)
        nov_path =0 # no. of vertices on the connectivity path
        for i in range(0,nov+1):
            if A[nov,i]>-1:
                B[nov_path]=A[nov,i]
                nov_path += 1
        B = B[0:nov_path]
        for i in range(0, B.size-1):
            plt.plot([vertices[0,int(B[i])],vertices[0,int(B[i+1])]],[vertices[1,int(B[i])],vertices[1,int(B[i+1])]],c='yellow',linewidth=7,alpha=0.5)
    elif flag ==0:
        flag = 0
        print('No solution has been found for the given maximum number of iterations.')
    else:
        print('The initial and goal configurations are close enough.')
        plt.plot([x0,x_goal],[y0,y_goal],c='yellow',linewidth=7,alpha=0.5)
        

    plt.savefig("parralel_tree"+str(num_tree+1)+".png")
    #plt.show()

#for i in range(10):
#    main()

"""
@cuda.jit
def kernel_euler_method(d_diff, d_f, d_stencil):
    i= cuda.grid(1)
    n= d_diff.shape[0]
    d_diff[0] = 1
    d_diff[i] = d_f[i] - d_f[i]*(d_f[i]-d_f[i-1])*d_stencil[0]

def wrap_fun_diff(f, dx):
    n=f.shape[0]
    c=1.0
    dt=0.02
    stencil = np.array([(dt/dx)])
    d_stencil = cuda.to_device(stencil)
    d_f = cuda.to_device(f)
    d_diff = cuda.device_array_like(d_f)
    blocks, threads = (n+TPB-1)//TPB, TPB
    kernel_euler_method[blocks, threads](d_diff, d_f, stencil)
    return d_diff.copy_to_host()

"""
"""
@cuda.jit
def kernel_para_RRT(d_diff):
    i= cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    if i < d_diff.shape[0]:
        d_diff[i] = 1
        #main()

start = cuda.event()
stop = cuda.event() 
starts = []
events = 10
for i in range(events):
    starts.append(cuda.event())

streams = []
for i in range(events):
    n = np.ones(10)
    num = 10
    #d_f = cuda.to_device(n)
    d_diff = cuda.device_array_like(n)
    streams.append(cuda.stream)
    starts[i].record(streams[-1])
    blocks, threads = (num+TPB-1)//TPB, TPB
    kernel_para_RRT[blocks, threads, streams[-1]](d_diff)
    main()
    stop.record()
    tot_time = cuda.event_elapsed_time(start, stop)
    print(" Time taken for RRT: ", i, "is: ", tot_time,"ms")
    #synchronize()

"""

#for s in streams:
#    s.synchronize()

#for i in events:
#    cuda.event_elapsed_time(start[i], end[i])
    #main()

import dask
import timeit
import time

from dask import delayed

main = delayed(main)
for n in range(16):
    start = time.time()
    _ = main(n).compute()
    tot_time = (time.time()-start)
    print("Time required for calculating tree : ", n+1, "is: ", tot_time, " sec")