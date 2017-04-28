# Parallelized Image Recognition in Spark + MPI
## CS205 - Final Report
## Tim Clements, Daniel Cusworth, Joannes (Bram) Maasakkers

Image recognition is an old subject in artificial intelligence, which has recently been subject to intense development thanks to computational advances. Private industry has developed a variety of multi-layered image recognition processors algorithms (e.g., AlexNet, GoogleNet) and software (e.g., TensorFlow, Caffe) to classify arbitrary images into categories. These advances have applications in facial recognition, self-driving automobiles, and social media (to name just a few). However, training large batches of images for classification is still a computationally intensive process and the predictive ability of an image recognition framework improves with the number of ingested samples.

Our project is to apply statistical learning theory in the MPI, OpenMP, and Spark frameworks to classify images. We further perform parallal hybridization of OpenMP and MPI. We develop three parallel algorithm frameworks, 1) model parallelism MPI + OpenMP, 2) data parallelism MPI + OpenMP, and 3) model parallelism Spark. 

### Learning algorithm
We implement a multi-class linear classifier (one hidden layer neutral network) to perform a training, validation, and testing split on the data. The learning algorithm is known as Regularized Linear Least Squares or Ridge Regression [(Tibshirani, 1996)].


The solution to the fitted "weights" or coefficients can be solved analytically:


Deep machine learning algorithms (e.g., GoogleNet) have several "layers" where more weights are fit with a nonlinear activation function. Thus solving the problem requires numerical optimization (e.g., stochastic gradient descent). For this pro

The current implementation uses the MNIST database [(LeCun et al. 1998)](http://yann.lecun.com/exdb/mnist/) that consists handwritten digits (0-9). The database includes a training set of 60,000 and a test set of 10,000 each consisting of 28 by 28 pixels. Each pixel has a value between 0 and 255 (white to black). If time permits, we are planning to expand our model to import our own images (see 'Future Work'). 

### Serial implementation
We first benchmark the serial implementation on Odyssey for different problem sizes. The code used for this is included in [Code_Serial.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Serial.py). As shown in the Figure below, we find that runtime scales linearly with the number of samples studied. The model reaches above 70% accuracy for the larger training sets. 

![Serial-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Sizes_Serial.png)

### OpenMP parallelization
OpenMP parallelization is implemented using Cython. The inner loops of the learning algorithms are parallelized using nogil pranges that  use static scheduling. By just parallelizing the inner loops using OpenMP, the outer loops can later be parallelized using MPI. The code is compiled using GCC (5.2.0) and benchmarked for different numbers of cores. We find that using Cython with just one thread (so no parallelization) already gives a large speedup compared to the regular serial code. For all problem sizes, we find speedups around 66 compared to the serial code. This speedup is due to ability of Cython to run in c. The figure below shows runtimes for different problem sizes (number of samples *n*) and different numbers of cores.

![OpenMP-Runtimes](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Runtime_OpenMP.png)

The Figure shows lower runtimes for an increasing number of cores. The Figure below shows the associated speedup and scaled speedup using *n = 1000* as the base case. We find that using both 5 and 10 cores leads to regular speedup above 1. Using 10 cores is slower than using 5 cores due to additional overhead (this is only true for *n = 1000*). However, when looking at the scaled speedup (increasing the problem size with the same ratio as increasing the number of processors), we do find that using 10 cores has the highest speedup. The brown line shows the efficiencies obtained with the different numbers of cores. Efficiencies are relatively small as we use a number of different pranges, each with their own overhead/allocation costs. As the problem size is relatively small, these overheads are relatively large. In addition to that, part of the module is not parallelized, leaving part of the code to be run in serial in all cases. The used Python script is [Code_OpenMP.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_OpenMP.py) together with Cython module [train_ml_prange.pyx](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/train_ml_prange.pyx).

![OpenMP-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_OpenMP.png)

### OpenMP + MPI parallelization
On top of the inner loop parallelization using OpenMP, we now implement MPI parallelization on the outer loop. This is implemented using the mpi4py package. We are currently working out some issues with the communication between the different nodes, benchmark results will be added shortly. The current associated Cython module is [train_ml_MPI.pyx](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/train_ml_MPI.pyx).

### Spark parallelization
Spark allows a different method of parallelizing the learning algorithm. Using functional parallelism, Spark parallelizes using compositions of functions. We implement a Spark version of our code on the Amazon Web Services (AWS) EMR Spark cluster. We run our code with 1 master and 2 core nodes and validate that it gives the same results as the serial implementation. The resulting speedup compared to running the serial Odyssey code is shown in the figure below.

![Spark-Speedups](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Speedup_Spark.png)

We find speedup larger than 1 for all problem sizes studied. More work will be done on applying Spark to the loop over the Tikhonov regularization factors and benchmarking it for varying hardware setups on AWS. The Spark version of the code is [Code_Spark.py](https://github.com/jdmaasakkers/cs205_prelimreport/blob/master/Code/Code_Spark.py). Python code used for all the plots is included in the code directory. 

### Future work
We will continue work along three different avenues:
- Optimize the hybrid parallelization using OpenMP + MPI and do a more rigorous benchmark including running on 8 compute nodes on Odyssey.
- Optimize the Spark parallelization for AWS (adding Spark to additional loops in the program), implement GPU acceleration, and benchmark for different setups. We will also add visuals illustrating the workflow. 
- Build a framework where custom images can be imported into the learning algorithm. 
