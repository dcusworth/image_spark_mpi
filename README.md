# Parallelized Image Recognition in Spark + MPI
## CS205 - Final Report
## Tim Clements, Daniel Cusworth, Joannes (Bram) Maasakkers

Image recognition is an old subject in artificial intelligence, which has recently been subject to intense development thanks to computational advances. Private industry has developed a variety of multi-layered image recognition processors algorithms (e.g., AlexNet, GoogleNet) and software (e.g., TensorFlow, Caffe) to classify arbitrary images into categories. These advances have applications in facial recognition, self-driving automobiles, and social media (to name just a few). However, training large batches of images for classification is still a computationally intensive process and the predictive ability of an image recognition framework improves with the number of ingested samples.

Our project is to apply statistical learning theory in the MPI, OpenMP, and Spark frameworks to classify images. We further perform parallal hybridization of OpenMP and MPI. We develop three parallel algorithm frameworks, 1) model parallelism MPI + OpenMP, 2) data parallelism MPI + OpenMP, and 3) model parallelism Spark. 

### Learning algorithm
Deep machine learning algorithms (e.g., GoogleNet) have several "layers" where more weights are fit with a nonlinear activation function. Thus solving the problem requires numerical optimization (e.g., stochastic gradient descent). For this pro

We implement a multi-class linear classifier (one hidden layer neutral network) to perform a training, validation, and testing split on the data. The learning algorithm is known as Regularized Linear Least Squares or Ridge Regression [(Tibshirani, 1996)](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf).

<center>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn1.png" alt="eqn1" style="width: 150px;"/>
</center>

Each row of X represents the pixels of an image. Each value of Y is the corresponding label of that image. The solution to the fitted "weights" or coefficients can be solved analytically:

<center>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn2.png" alt="eqn2" style="width: 150px;"/>
</center>

Where the pseudo-interve is definted as 

<center>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn3.png" alt="eqn3" style="width: 150px;"/>
</center>

For a multiclass classification of k labels, we need to solve for the analytical solution for each k class, where each image is classified as "1" when the label equals k, and "-1" otherwise. 

Following Bayes Decision Rule, we arrive at a prediction of being in or outside class k by looking at the sign of the prediction <X, w>. We decide the prediction among classes by solving the following:

<center>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn4.png" alt="eqn4" style="width: 150px;"/>
</center>


 
We train our classifier on the commonly used MNIST database [(LeCun et al. 1998)](http://yann.lecun.com/exdb/mnist/) that consists handwritten digits (0-9) (Figure 1). The database includes a training set of 60,000 and a test set of 10,000 each consisting of 28 by 28 pixels. Each pixel has a value between 0 and 255 (white to black).

We also implement a classifier of images we took ourselves of hands (Figure 1), which digits of 0-5. 

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/data.png" alt="data" style="width: 150px;"/>
<figcaption> Figure 1: Datasets used in this study. </figcaption>
</figure>



### Computation Graphs
We translate our learning algorithm to a computation graph (Figure 2). The analytical solution requires solving l pseudo-inverses for the length of the regularization (i.e., lambda grid). This presents us with an opportunity to perform model parallelism.


<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_1.png" alt="dag1" style="width: 150px;"/>
<figcaption> Figure 2: Computation graph for model parallelism. </figcaption>
</figure>

*Model Parallelism MPI + OpenMP*: We assign to each node a value of lambda, and have it compute the pseudo inverse, analytical solution, and classification for that value of each lambda. The MPI (Python package mpi4py) then communicates across nodes to see which lambda gives the best accuracy on a randomly reserved validation set of images and chooses that lambda as the optimal version of the model. We further parallelize the matrix multiplications in the analytical solution using OpenMP in Cython.

*Spark inner-loop*: We perform the innermost matrix multiplications using Spark by looping over each lambda and label, and treating the pseudo-inverse as an RDD, and multiplying it with X^TY.  

*Spark outer-loop*: We treat each lambda as an RDD, then send to the workers a lambda and the broadcasted X feature training set. This parallelizes the calculation of the pseudo-inverse. 


We can think of parallelism in a data framework as well (Figure 3).

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_2.png" alt="dag2" style="width: 300px;"/>
<figcaption> Figure 3: Computation graph for data parallelism. </figcaption>
</figure>

*Data Parallelism MPI + OpenMPI*. We compute the computation graph as in Figure 2, but for a subset of the data, which are sent to MPI nodes. After each node estimates the weights on that subset, the weights are brought together and averaged becfore making a prediction on the validation set.



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
We implement a Spark version of our code on an Amazon Web Services (AWS) EMR cluster (m2xlarge) using 1 master and 4 worker cores. Figure XX shows the results for both outer and inner parallelism ([Code listing for Spark-outer](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_spark_outer.py)) ([Code listing for Spark-inner](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_spark_inner.py)) ([Code listing for serial implementation](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_serial.py)). We see around 7x speedup for the outer loop Spark implementation. The inner loop implementation runs nearly the same as the serial code. We hypothesize that this is due to the fact that the MNSIT dataset's pixel dimension is low, meaning that the parallelization from just inner-most matrix multiplication provides little speedup over the serial version. However, the outer-loop implementation matches nicely with the model parallel results of MPI+OpenMP. 
 
<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/spark_speedup.png" alt="spark" style="width: 300px;"/>
<figcaption> Figure 3: Computation graph for data parallelism. </figcaption>
</figure>


We were only able to run for 20,000 images in the MNIST dataset, as the outer loop Spark code ran out of memory. 


### Future work
We will continue work along three different avenues:
- Optimize the hybrid parallelization using OpenMP + MPI and do a more rigorous benchmark including running on 8 compute nodes on Odyssey.
- Optimize the Spark parallelization for AWS (adding Spark to additional loops in the program), implement GPU acceleration, and benchmark for different setups. We will also add visuals illustrating the workflow. 
- Build a framework where custom images can be imported into the learning algorithm. 
