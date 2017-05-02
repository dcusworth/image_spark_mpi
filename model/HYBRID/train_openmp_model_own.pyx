#cython: boundscheck=False, wraparound=False, nonecheck=False

import cython
from cython.parallel import prange, parallel
import numpy as np
cimport numpy as np
import random

DTYPE64 = np.float64
DTYPE32 = np.int
ctypedef np.int_t DTYPE32_t
ctypedef np.float64_t DTYPE64_t

def label_func(x, choose_label):
	if x == choose_label:
		return 1
	else:
		return -1

def train(np.ndarray[np.float64_t, ndim=2] train_data, 
	      np.ndarray[np.float64_t, ndim=2] test_data, 
	      np.ndarray[np.float64_t, ndim=2] denom_noninv,
		  np.ndarray[np.int64_t, ndim=1] train_labels, 
		  np.ndarray[np.int64_t, ndim=1] test_labels, 
		  float ll,
		  int num_threads):
	"""
	Train regression on image data using cython using Model Parallelism. 


	Inputs
	------
	:type train_data: numpy.ndarray (of numpy.float64)
	:param train_data: Training images 
	:type test_data: numpy.ndarray (of numpy.float64)
	:param test_data: Testing images 
	:type denom_noninv: numpy.ndarray (of numpy.float64)
	:param denom_noninv: Product of train_data.T * train_data
	:type train_labels: list (of ints)
	:param train_labels: digit (0-9) for each training image
	:type test_labels: list (of ints)
	:param test_labels: digit (0-9) for each testing image
	:type ll: numpy.float64
	:param ll: regularization parameter
	:type num_threads: int
	:param num_threads: Number of OpenMP threads

	Returns
	-------
	:type vacc: np.float
	:param vacc: Testing validation accuracy (0 to 1) 
	:type preds: List (of ints)
	:param preds: Predictions for each image in data set 

	Written by Tim Clements, Dan Cusworth, and Bram Maasakkers
	Last modified 5/1/17

	"""
	
	cdef int reg, choose_label, q, i, j, idx, p, N = 6
	cdef int Nt = train_data.shape[0],Mt = train_data.shape[1]
	cdef int Ntest = test_data.shape[0],Mtest = test_data.shape[1]
	cdef float idenom, vacc, topenmp, start_tp, end_tp, start_loop,start_all, tp=0.,t
	cdef np.int64_t imost
	cdef np.ndarray[np.float64_t, ndim=2] x_T = np.empty([Mt,Nt], dtype=DTYPE64)
	cdef np.ndarray[np.int64_t, ndim=1] y_tr_map = np.empty(Nt, dtype=DTYPE32)
	cdef np.ndarray[np.float64_t, ndim=1] numer_sum = np.empty(Mt,dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=1] x_iT = np.empty_like(numer_sum)
	cdef np.ndarray[np.float64_t, ndim=1] inumer = np.empty_like(numer_sum)
	cdef np.ndarray[np.float64_t, ndim=1] iw = np.empty_like(numer_sum)
	cdef np.ndarray[np.float64_t, ndim=1] iout = np.empty(test_data.shape[0],dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=1] iclass = np.empty_like(iout)
	cdef np.ndarray[np.float64_t, ndim=2] ws = np.empty([N,Mt], dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] iouts = np.empty([N,test_data.shape[0]],
														 dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] iclasses = np.empty([N,test_data.shape[0]],
														 dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=1] ipred = np.empty_like(iout)
	cdef np.ndarray[np.float64_t, ndim=1] preds = np.empty(test_data.shape[0],dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] denom_sum = np.empty([Mt,Mt],dtype=DTYPE64)

	# Add regularization parameter
	denom_noninv += ll*np.eye(Mt)
	denom_sum = np.linalg.inv(denom_noninv)

	# Loop over each digit in data set
	for choose_label in range(N): 
		for i in range(Nt):
			y_tr_map[i] = label_func(train_labels[i],choose_label)

		# Sum of each pixel 
		numer_sum = np.zeros(Mt,dtype=DTYPE64)
		iw = np.zeros(Mt)
		iout = np.zeros(test_data.shape[0],dtype=DTYPE64)
		for i in prange(Nt,nogil=True,schedule='static',num_threads=num_threads):
			for j in range(Mt):
				numer_sum[j] += train_data[i,j] * y_tr_map[i]
			
		# Compute weights (X^T*X + lambda*I) * Sumj(Train_data)
		for i in prange(Mt,nogil=True,schedule='static',num_threads=num_threads):
			for j in range(Mt):
				iw[i] += denom_sum[i,j] * numer_sum[j]

		# Apply weights to test data
		for i in prange(Ntest,nogil=True,schedule='static',num_threads=num_threads):
			for j in range(Mtest):
				iout[i] += test_data[i,j] * iw[j] 

		for i in range(len(iout)):
			iclass[i] = np.sign(iout[i])


		# Append to output
		ws[choose_label,:] = iw
		iouts[choose_label,:] = iout
		iclasses[choose_label,:] = iclass
 
	preds = np.zeros(Ntest,dtype=DTYPE64)
	for idx in range(Ntest):
		ipreds = iouts[:,idx]
		imost = np.where(ipreds == np.max(ipreds))[0][0] 
		preds[idx] = imost

	# Determine accuracy on validation
	vacc = 0.
	for i in range(len(test_labels)):
		if test_labels[i] == preds[i]:
			vacc += 1
	vacc /= float(len(preds))
	return vacc, preds

cpdef matmat_multi_tile(np.ndarray[np.float64_t, ndim=2] A, 
	                    np.ndarray[np.float64_t, ndim=2] B,
	                    int num_threads):
	"""
	Tiled matrix-multiplication using Cython and OpenMP

	Inputs
	------
	:type A: numpy.ndarray (of numpy.float64)
	:param A: NxM matrix 
	:type B: numpy.ndarray (of numpy.float64)
	:param B: MXN matrix 
	:type num_threads: int
	:param num_threads: Number of OpenMP threads

	Returns
	-------
	:type Cmat: numpy.ndarray (of numpy.float64)
	:param vacc: Matrix product C = A*B 

	Written by Tim Clements, Dan Cusworth, and Bram Maasakkers
	Last modified 5/1/17

	"""
	cdef int n = A.shape[0],m = A.shape[1], b = 100, b_two = 100
	cdef np.ndarray[np.float64_t, ndim=2] Cmat = np.empty([n,n], dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] Ablock = np.empty([b,b_two], dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] Bblock = np.empty([b_two,b], dtype=DTYPE64)
	cdef np.ndarray[np.float64_t, ndim=2] Cblock = np.empty([b,b], dtype=DTYPE64)
	cdef int outer_i, outer_j, outer_k, i, j, k 

	# Outer loops that iterate over size of block b
	for outer_i in range(0,n,b):
		for outer_j in range(0,n,b):
			Cblock = np.zeros([b,b]) #Read Cblock into fast memory		
			
			# Select blocks
			for outer_k in range(0,m,b_two):

				# A to fast memory	
				Ablock = np.asarray(A[outer_i:(outer_i+b), outer_k:(outer_k+b_two)])
				# B to fast memory
				Bblock = np.asarray(B[outer_k:(outer_k+b_two), outer_j:(outer_j+b)])
				
				# Perform matrix multiplication on these blocks
				for i in prange(b, nogil=True, schedule='static', num_threads=num_threads):
					for j in range(b):
						for k in range(b_two):
							Cblock[i,j] += Ablock[i,k] * Bblock[k,j]

			# Write to slow memory
			Cmat[outer_i:(outer_i+b), outer_j:(outer_j+b)] = Cblock
	return Cmat