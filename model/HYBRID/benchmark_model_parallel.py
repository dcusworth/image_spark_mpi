import pyximport
pyximport.install()
from mpi4py import MPI
import numpy as np
import time
import random
import sys

from mnist import MNIST
import train_mpi_model

""" Model parallel image recogntion benchmark for MNIST and finger data sets.

Uses MPI and OpenMP hybrid parallelism. Calls train_mpi_model.pyx Cython 
subroutine. 

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

########### MAKE DATA SELECTION ###########
num_lambda = 10 # Number of individual lambdas
N = 10000 # Number of images
nthreads = int(sys.argv[1]) # number of OpenMP threads
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
			if which_data == 'MNIST':
				mndata = MNIST('data')
				images,labels = mndata.load_training()
		comm.Barrier()
else: # size == 0
	if which_data == 'MNIST':
		mndata = MNIST('data')
		images,labels = mndata.load_training()

# distribute regularization parameters among ranks 
if rank == 0:
	all_preds = []
	allT = []
	lambdas = np.array([10**q for q in np.linspace(-5,5,num_lambda)])
	vaccs = np.empty_like(lambdas)
	if size > 1: # split data into number of ranks
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
if rank == 0:
	start = time.time()
	x_T = Xtr.T
	start_tp = time.time()
	denom_noninv = train_mpi_model.matmat_multi_tile(x_T,Xtr,nthreads) 
	tp = time.time() - start_tp
else:
	denom_noninv = None
denom_noninv = comm.bcast(denom_noninv,root = 0)

# Begin model parallelism section 
start_train = time.time()
for ll in lambdas:
	#### CALL cython implementation ####
	vacc, preds = train_mpi_model.train(Xtr, Xvl,denom_noninv,y_tr, y_val,ll,nthreads)
	#### END cython implementation ####
t_mpi = time.time() - start_train

# gather everything  
if size > 1:
	comm.Barrier()
	vacc = comm.gather(vacc,root=0)
	lambdas = comm.gather(lambdas,root=0)
else:
	vacc = np.array([vacc])
	lambdas = np.array([lambdas])

# print results 
if rank == 0:
	t_tot = time.time() - start
	print('#############################################')
	print('matrix mult {}'.format(tp))
	best_val = np.where(vacc == np.max(vacc))[0][0]
	print('validation accuracy = {}'.format(vacc[best_val]))
	print('best lambda = {}'.format(lambdas[best_val]))
	print('{} samples, {} nodes, {} threads, t total {} seconds, tMPI {} seconds'.format(N,size,nthreads,t_tot,t_mpi))
