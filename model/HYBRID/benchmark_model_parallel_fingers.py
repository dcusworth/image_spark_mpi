import pyximport
pyximport.install()
from mpi4py import MPI
import numpy as np
import time
import random
import sys
import matplotlib.image as mpimg
import scipy.interpolate
import glob

import train_openmp_model_own

""" Model parallel image recogntion benchmark for finger image data set.

Uses MPI and OpenMP hybrid parallelism. Calls train_openmp_model_own.pyx Cython 
subroutine. Downsamples .png images into more mangeable size to fit images 
into memory. 

Parameters
----------
num_lambda : Number of regularization parameters
N          : Number of MNIST images 
nthreads   : Number of OpenMP threads
size       : Number of MPI ranks

Prints
-------
t_tot : Total running time (in seconds) for OpenMP + MPI
t_mp  : Running time (in seconds) for OpenMP loops 

Written by Tim Clements, Dan Cusworth, and Bram Maasakkers
Last modified 5/1/17
"""

def load_own():
	"""
	Reads images and labels from finger images dataset.

	Written by Tim Clements, Dan Cusworth, and Bram Maasakkers
	Last modified 5/1/17
	"""
	images_in = []
	labels_in = []

	#Read 6 classes
	for fingers in np.arange(6):
		list_images = glob.glob("/n/regal/iacs/image_spark_mpi/Own_Data/class_"+str(fingers)+"/*.png")

		for i in np.arange(len(list_images)):

			#Read image
			single_image = np.array(mpimg.imread(list_images[i])[:,:,0]) #Select only layer 1

			#Make full binary
			single_image[single_image<0.5] = 0.00
			single_image[single_image>0.5] = 1.00

			#Interpolate
			xdim = np.linspace(0,180,180)
			ydim = np.linspace(0,120,120)
			interp_func = scipy.interpolate.interp2d(xdim, ydim, single_image)
			interp_new = interp_func(np.linspace(0,180,60), np.linspace(0,120,40))
			interp_new[interp_new < 0.5] = 0
			interp_new[interp_new > 0.5] = 1

			#Reshape to 1D, add image & label to list
			single_image = list(np.reshape(interp_new,[60*40]))
			images_in.append(single_image)
			labels_in.append(fingers)

	images = []
	labels = []
	index_shuf = list(range(len(labels_in)))
	random.shuffle(index_shuf)
	for i in index_shuf:
		images.append(images_in[i])
		labels.append(labels_in[i])
	
	return images, labels

########### MAKE DATA SELECTION ###########
num_lambda = 10 # Number of individual lambdas
N = 16000 # Number of images
nthreads = int(sys.argv[1]) # number of OpenMP threads
d = 2400 # Pixels of our own data
###########################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('###################')
print('Running with {}'.format(nthreads))
print('###################')

# loop through ranks to prevent multiple data access
if size > 1:
	for ii in range(size):
		if rank == ii:
			images,labels = load_own()
		comm.Barrier()
else: # size == 0
	images,labels = load_own()

# distribute regularization parameters among ranks
if rank == 0:
	all_preds = []
	allT = []
	lambdas = np.array([10**q for q in np.linspace(-5,5,num_lambda)])
	vaccs = np.empty_like(lambdas)
	if size > 1:
		lambda_split = np.array_split(lambdas,size)
else:
	lambda_split = None

# Construct training and testing data sets
if size > 1:
	lambdas = comm.scatter(lambda_split,root=0)
images,labels = np.array(images),np.array(labels)
images = images[0:N,:]
labels = labels[0:N]
tint = int(.8*N)

# Remove bias from images
fmean = images.mean(axis=0)
x_c = images - fmean[np.newaxis,:]
Xtr = x_c[0:tint]
Xvl = x_c[tint:-1]
y_val = labels[tint:-1].astype(np.int)
y_tr = labels[0:tint].astype(np.int)


# Compute X^T * X on rank 0, distribute among ranks
if size > 1:
	comm.Barrier()

if rank == 0:
	start = time.time()
	x_T = Xtr.T
	start_tp = time.time()
	denom_noninv = train_openmp_model_own.matmat_multi_tile(x_T.astype(np.float64),Xtr.astype(np.float64),nthreads) 
	tp = time.time() - start_tp
else:
	denom_noninv = None

comm.Barrier()
denom_noninv = comm.bcast(denom_noninv,root = 0)

# Begin model parallelism section
start_train = time.time()
for ll in lambdas:
	#### CALL cython implementation ####
	vacc, preds = train_openmp_model_own.train(Xtr, Xvl,denom_noninv,y_tr, y_val,ll,nthreads)
	#### END cython implementation ####
t_mpi = time.time() - start_train

# gather everything  
if size > 1:
	comm.Barrier()
	vacc = comm.gather(vacc,root=0)
	lambdas = comm.gather(lambdas,root=0)
	# t = comm.gather(t,root=0)
else:
	vacc = np.array([vacc])
	lambdas = np.array([lambdas])

if rank == 0:
	t_tot = time.time() - start
	print('#############################################')
	print('matrix mult {}'.format(tp))
	best_val = np.where(vacc == np.max(vacc))[0][0]
	print('validation accuracy = {}'.format(vacc[best_val]))
	print('best lambda = {}'.format(lambdas[best_val]))
	print('{} samples, {} nodes, {} threads, t total {} seconds, tMPI {} seconds'.format(N,size,nthreads,t_tot,t_mpi))
