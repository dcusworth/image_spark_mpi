# Parallelized Image Recognition in Spark + MPI
## CS205 - Final Report
## Tim Clements, Daniel Cusworth, Joannes (Bram) Maasakkers

Image recognition is an old subject in artificial intelligence, which has recently been subject to intense development thanks to computational advances. Private industry has developed a variety of multi-layered image recognition processors algorithms (e.g., AlexNet, GoogleNet) and software (e.g., TensorFlow, Caffe) to classify arbitrary images into categories. These advances have applications in facial recognition, self-driving automobiles, and social media (to name just a few). However, training large batches of images for classification is still a computationally intensive process and the predictive ability of an image recognition framework improves with the number of ingested samples.

Our project is to apply statistical learning theory in the MPI, OpenMP, and Spark frameworks to classify images. We further perform parallal hybridization of OpenMP and MPI. We develop three parallel algorithm frameworks, 1) model parallelism MPI + OpenMP, 2) data parallelism MPI + OpenMP, and 3) model parallelism Spark. 

### Learning algorithm
Deep machine learning algorithms (e.g., GoogleNet) have several "layers" where weights are fit with a nonlinear activation function. Thus solving the problem requires numerical optimization (e.g., stochastic gradient descent). For this project we implement a multi-class linear classifier (one hidden layer neutral network) to perform a training, validation, and testing split on the data. The learning algorithm is known as Regularized Linear Least Squares or Ridge Regression [(Tibshirani, 1996)](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf).

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn1.png" alt="eqn1" WIDTH="300"/>

Each row of X represents the pixels of an image. Each value of Y is the corresponding label of that image. The solution to the fitted "weights" or coefficients can be solved analytically:

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn2.png" alt="eqn2" WIDTH="250"/>

Where the pseudo-inverse is definted as 

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn3.png" alt="eqn3" WIDTH="150"/>

For a multiclass classification of k labels, we need to solve for the analytical solution for each k class, where each image is classified as "1" when the label equals k, and "-1" otherwise. 

Following Bayes Decision Rule, we arrive at a prediction of being in or outside class k by looking at the sign of the prediction (X*w). We decide the prediction among classes by solving the following:

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn4.png" alt="eqn4" WIDTH="500"/>


 
We train our classifier on the commonly used MNIST database [(LeCun et al. 1998)](http://yann.lecun.com/exdb/mnist/) that consists handwritten digits (0-9) (Figure 1). The database includes a training set of 60,000 and a test set of 10,000 each consisting of 28 by 28 pixels. Each pixel has a value between 0 and 255 (white to black).

We also implement a classifier of images we took ourselves of hands (Figure 1), which digits of 0-5. 

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/data.png" alt="data" WIDTH="300"/>
<figcaption> Figure 1: Datasets used in this study - MNIST (left), and our own photographed images (right). </figcaption>
</figure>
<br>



### Computation Graphs
We translate our learning algorithm to a computation graph (Figure 2). The analytical solution requires solving l pseudo-inverses for the length of the regularization (i.e., lambda grid). This presents us with an opportunity to perform model parallelism.


<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_1.png" alt="dag1" WIDTH="500"/>
Figure 2: Computation graph for model parallelism.



*Model Parallelism MPI + OpenMP*: We assign to each node a value of lambda, and have it compute the pseudo inverse, analytical solution, and classification for that value of each lambda. The MPI (Python package mpi4py) then communicates across nodes to see which lambda gives the best accuracy on a randomly reserved validation set of images and chooses that lambda as the optimal version of the model. We further parallelize the matrix multiplications in the analytical solution using block-tiling in OpenMP.

*Spark inner-loop*: We perform the innermost matrix multiplications using Spark by looping over each lambda and label, and treating the pseudo-inverse as an RDD, and multiplying it with X^TY.  

*Spark outer-loop*: We treat each lambda as an RDD, then send to the workers a lambda and the broadcasted X feature training set. This parallelizes the calculation of the pseudo-inverse. 


We can think of parallelism in a data framework as well (Figure 3).

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_2.png" alt="dag2" WIDTH="500"/>
Figure 3: Computation graph for data parallelism.


*Data Parallelism MPI + OpenMPI*. We compute the computation graph as in Figure 2, but for a subset of the data, which are sent to MPI nodes. After each node estimates the weights on that subset, the weights are brought together and averaged becfore making a prediction on the validation set.

We run our hybrid MPI-OpenMP code on the RC Odyssey cluster. RC Odyssey is large-scale, heterogeneous computing facility run at two locations in Boston, MA and one in Holyoke, MA. RC Odyssey has over 65,000 cores, 260 TB of RAM and over 1,000,000 CUDA cores. We use the seas_iacs partition of Odyssey. Each node on the seas_iacs partition has 4 sockets, with 8 cores per socket and 2 threads per core, for a total of 64 CPUs/nodes. The CPUs are x86_64 AMD Opteron 6376 Processors, each of which runs at 2300 MHz and has 4 Gb of RAM. 

### Results 

In all parallel configurations shown below, we achieve 85% prediction accuracy using the MNIST dataset.

**Amdahl's Law**

We first analyze our code to understand the degree to which the matrix multiplications can be parallelize. The solution to fitting the weights during training is written as

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn2.png" alt="eqn2" WIDTH="300"/>

During testing, we then use these weights to compute

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn9.png" alt="eqn9" WIDTH="300"/>

Together, the number of computations is 

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn11.png" alt="eqn11" WIDTH="300"/>


We note that since  k << d, the Nd^2 term is going to dominate the computation. To see how much our problem can be parallelized, we time how long the parallelizable portion of the code runs, and the time that the code's overhead takes to run. We run on one node, and vary the number of threads from 1 to 16. Thus the time to run on a single node can be written as

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn12.png" alt="eqn12" WIDTH="300"/>

The results of running on several cores for 40,000 images are shown below in Figure (XX). As more threads are added, the parallel component of the algorithm speeds up until reaching 8 cores, at which it stabalizes. The overhead component begins to slightly increase as threads are added, indicating the increased communication that comes with adding more processors. These results show up that for a problem size of 40,000 images, we achieve maximum OpenMP parallelization betwen 4 and 8 cores.

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/amdahl.png" alt="amdahl" WIDTH="300";"/>
<figcaption> Figure XX: Parallel speedup on a single node varying threads, broken in overhead and parallelizable components. </figcaption>
</figure>
<br>


*Hybrid OpenMP + MPI - Model Parallelism*

Using the model parallel framework described above (OpenMP on matrix multiplications, MPI on lambdas), we achieve the following results (Figure XX) when varying threads and nodes. We see the maximum speedup occuring with the maximum number of nodes and threads (8 each). The efficiency drops as we increase the threads and nodes, but the scaled speedup is still largest number of threads and nodes.

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/model_hybrid.png" alt="model_par" WIDTH="300";"/>
<figcaption> Figure XX: Speedup, Scaled Speedup, and Efficiency for hybrid model parallelism. </figcaption>
</figure>
<br>

*Hybrid OpenMP + MPI - Data Parallelism*

Using the data parallel framework described above (OpenMP on matrix multiplications, MPI on subsets of the images), we achieve the following results (Figure XX) when varying threads and nodes. Like with the model parallel framework, we see maximum speeup for the maximum number of threads and nodes requested. However, the speedups are much larger in the data parallel framework than the model parallel framework (25x versus 4x, respectively). We also see much better efficiency in the data parallel approach, where the efficiency remains near optimal for many thread, node configurations.

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/data_hybrid.png" alt="data_par" WIDTH="300";"/>
<figcaption> Figure XX: Speedup, Scaled Speedup, and Efficiency for hybrid data parallelism. </figcaption>
</figure>
<br>


*Spark parallelization*
We implement a Spark version of our code on an Amazon Web Services (AWS) EMR cluster (m2xlarge) using 1 master and 4 worker cores. Figure XX shows the results for both outer and inner parallelism ([Code listing for Spark-outer](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_spark_outer.py)) ([Code listing for Spark-inner](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_spark_inner.py)) ([Code listing for serial implementation](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_serial.py)). We see around 7x speedup for the outer loop Spark implementation. The inner loop implementation runs nearly the same as the serial code. We hypothesize that this is due to the fact that the MNSIT dataset's pixel dimension is low, meaning that the parallelization from just inner-most matrix multiplication provides little speedup over the serial version. However, the outer-loop implementation matches nicely with the model parallel results of MPI+OpenMP. 
 
<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/spark_speedup.png" alt="spark" WIDTH="300"/>
<figcaption> Figure XX: Computation graph for data parallelism. </figcaption>
</figure>


We were only able to run Spark for 20,000 images in the MNIST dataset, as the outer loop Spark code ran out of memory. 


### Future work
We will continue work along three different avenues:
- Optimize the hybrid parallelization using OpenMP + MPI and do a more rigorous benchmark including running on 8 compute nodes on Odyssey.
- Optimize the Spark parallelization for AWS (adding Spark to additional loops in the program), implement GPU acceleration, and benchmark for different setups. We will also add visuals illustrating the workflow. 
- Build a framework where custom images can be imported into the learning algorithm. 
