import pyximport
pyximport.install()
from mpi4py import MPI
import numpy as np
import time
import random
import sys

from mnist import MNIST
import train_openmp_data

""" Data parallel image recogntion benchmark for MNIST image data set.

Uses MPI and OpenMP hybrid parallelism. Calls train_openmp_data.pyx Cython 
subroutine. 

Parameters
----------
num_lambda : Number of regularization parameters
N          : Number of MNIST images 
nthreads   : Number of OpenMP threads
size       : Number of MPI ranks

Prints
-------
t : Total running time (in seconds) for OpenMP + MPI 

Written by Tim Clements, Dan Cusworth, and Bram Maasakkers
Last modified 5/1/17
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

########### MAKE DATA SELECTION ###########
num_lambda = 10 # Number of individual lambdas
N = 10000 # Number of images
nthreads = int(sys.argv[1]) # number of OpenMP threads
lambdas = np.array([10**q for q in np.linspace(-5,5,10)])
vaccs = np.zeros_like(lambdas)
###########################################


# Load data on master rank, then split between all ranks 
if rank == 0:
	nthreads = sys.argv[1]
	mndata = MNIST('data')
	images,labels = mndata.load_training()
	images = images[0:N]
	labels = labels[0:N]
	split_images = np.array_split(images,size,axis=0)
	split_labels = np.array_split(labels,size,axis=0)
	vaccs = np.empty_like(lambdas)
	all_preds = []
	allT = []
else:
	split_images,split_labels, nthreads = None,None,None

images = comm.scatter(split_images, root=0)
labels = comm.scatter(split_labels, root=0)
nthreads = comm.bcast(nthreads,root=0)

# Construct training and testing data sets
N = images.shape[0]
sinds = range(N)
random.shuffle(list(sinds))
tint = int(.8*N)
tind = sinds[0:tint]
vind = sinds[tint:-1]

# Remove bias from images
fmean = images.mean(axis=0)
x_c = images - fmean[np.newaxis,:]
Xtr = x_c[0:tint]
Xvl = x_c[tint:-1]
y_val = np.array(labels[tint:-1]).astype(np.int)
y_tr = np.array(labels[0:tint]).astype(np.int)

# Make sure all ranks begin at same time
comm.barrier()
start = time.time()

# Compute X^T * X 
x_T = Xtr.T
denom_noninv = train_openmp_data.matmat_multi_tile(x_T,Xtr,int(nthreads))

# loop through regularization parameters
for ii,ll in enumerate(lambdas):
	iws = train_openmp_data.train(Xtr,denom_noninv,y_tr,ll,int(nthreads))
	
	# gather weights for each regularization and take average
	all_iw = comm.gather(iws,root = 0) 
	if rank == 0:
		iw = np.mean(all_iw,axis=0)
	else:
		iw = None
	iw = comm.bcast(iw,root = 0)

	# Test using averaged weights 
	vacc, preds = train_openmp_data.test(Xvl,y_val, iw, int(nthreads))
	
	# Gather validation accuracies and save average value
	vacc = comm.gather(vacc,root=0)
	if rank ==0:
		vacc = np.mean(vacc)
		vaccs[ii] = vacc

end = time.time()
t = end - start
comm.Barrier()
t = comm.gather(t,root=0)

# print results
if rank == 0:
	best_val = np.where(vaccs == np.max(vaccs))[0][0]
	print('#####################################################################')
	print('validation accuracy = {}'.format(vaccs[best_val]))
	print('best lambda = {}'.format(lambdas[best_val]))
	print('{} samples, {} nodes, {} threads, {} seconds'.format(N * size,size,nthreads,np.max(t)))
