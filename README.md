# Parallelized Image Recognition in Spark and MPI+OpenMP
## CS205 - Final Report
## Tim Clements, Daniel Cusworth, Joannes (Bram) Maasakkers

Image recognition is a classic topic in artificial intelligence, which has recently been subject to intense development thanks to computational advances. There are a variety of multi-layered image recognition algorithms (e.g., AlexNet, GoogleNet) and software (e.g., TensorFlow, Caffe) to classify arbitrary images into categories. These packages have applications in facial recognition, self-driving automobiles, and social media (to name just a few). However, training large batches of images for classification is still a computationally intensive process and the predictive ability of an image recognition framework improves with the number of trained samples.

Our project is to apply statistical learning theory in the MPI, OpenMP, and Spark frameworks to classify images. We further perform parallal hybridization of OpenMP and MPI. We develop three parallel algorithm frameworks, 1) model parallelism in MPI + OpenMP, 2) data parallelism in MPI + OpenMP, and 3) model parallelism Spark. 

### Learning algorithm
Deep machine learning algorithms have several layers where weights are fit after applying some nonlinear activation function. Thus solving the problem requires numerical optimization (e.g., stochastic gradient descent). For this project we implement a multi-class linear classifier (one hidden layer neutral network) to perform a training, validation, and testing on the data. The learning algorithm is known as Regularized Linear Least Squares or Ridge Regression [(Tibshirani, 1996)](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf).

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn1.png" alt="eqn1" WIDTH="350"/>

Each row of X represents the pixels of an image. Each value of Y is the corresponding label of that image. The solution to the fitted "weights" or coefficients can be solved analytically:

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn2.png" alt="eqn2" WIDTH="250"/>

The pseudo-inverse is defined as the following:

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn3.png" alt="eqn3" WIDTH="150"/>

For a multiclass classification of k labels, we solve the analytical solution for each k class, where each image is classified as "1" when the label equals k, and "-1" otherwise. 

Following the Bayes Decision Rule, we arrive at a prediction of being in or outside class k by looking at the sign of the multiplication (X*w). We decide the best prediction among classes by solving the following:

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn4.png" alt="eqn4" WIDTH="500"/>


 
We train our classifier on the commonly used MNIST database [(LeCun et al. 1998)](http://yann.lecun.com/exdb/mnist/) that consists handwritten digits (0-9) (Figure 1). Each image is 28 by 28 pixels and each pixel is a grayscale value between 0 and 255 (white to black). We do a 80%/20% train/test split of our data.

We also implement a classifier of images we took of our hands (Figure 1), with digits of 0-5. We filmed different test subjects in a studio setup and exported the individual frames of these movies to images of dimension 60 x 40 in black and white.  

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/data.png" alt="data" WIDTH="300"/>
<figcaption> Figure 1: Datasets used in this study - MNIST (left), and our own photographed images (right). </figcaption>
</figure>
<br>

 

### Computation Graphs
We translate the learning algorithm to a computation graph (Figure 2). The analytical solution requires solving l pseudo-inverses for the length of the regularization (i.e., lambda grid). We implement model parallelism to effeciently solve for each regularization parameter.


<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_1.png" alt="dag1" WIDTH="500"/>
Figure 2: Computation graph for model parallelism.
<br />  
 

*Model Parallelism MPI + OpenMP*: The most resource-intensive computation in the model parallel framework is computing the matrix multiplication X^T * X. The master node computes X^T * X using a block-tiling matrix multiplication routine, with shared memory and threads in OpenMP (Cython prange), then broadcasts the product to all nodes using MPI (Python package mpi4py). We then distribute to each node a subset of lambda values. Within that node, we compute the pseudo-inverse, analytical solution, and classification for each value lambda using OpenMP. The classification step applies weights from the training set to a randomly reserved validation set of images. The master nodes then gathers the validation accuracy from each lambda on all nodes. We then choose the lambda with the higest validation accuracy as the optimal version of the model. 
 
*Spark inner-loop*: We perform the innermost matrix multiplications using Spark by looping over each lambda and label, treating the pseudo-inverse as an RDD, and multiplying it with X^TY.  

*Spark outer-loop*: We treat each lambda as an RDD, then send to the workers a lambda and the broadcasted training set. This parallelizes the calculation of the pseudo-inverse. 


We can think of parallelism in a data framework as well (Figure 3).

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_2.png" alt="dag2" WIDTH="600"/>
Figure 3: Computation graph for data parallelism.
<br />  
 

*Data Parallelism MPI + OpenMPI*. We compute the computation graph as in Figure 2, but for a subset of the data, which are sent to MPI nodes. After each node estimates the weights on that subset, the weights are brought together and averaged before making a prediction on the validation set.

We run our hybrid MPI-OpenMP code on the RC Odyssey cluster. RC Odyssey is large-scale, heterogeneous computing facility run at two locations in Boston, MA and one in Holyoke, MA. RC Odyssey has over 65,000 cores, 260 TB of RAM and over 1,000,000 CUDA cores. We use the seas_iacs partition of Odyssey. Each node on the seas_iacs partition has 4 sockets, with 8 cores per socket and 2 threads per core, for a total of 64 CPUs/nodes. The CPUs are x86_64 AMD Opteron 6376 Processors, each of which runs at 2300 MHz and has 4 Gb of RAM.

We implement the Spark version of our code on an Amazon Web Services (AWS) EMR cluster (m2xlarge) using 1 master and 4 worker cores.

### Results 

In all parallel configurations shown below, we achieve 85% prediction accuracy using the MNIST dataset.

**Amdahl's Law**

We first analyze our code to understand the degree to which the matrix multiplications can be parallelized. The solution to fitting the weights during training is written as

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn2.png" alt="eqn2" WIDTH="250"/>

During testing, we then use these weights to compute

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn9.png" alt="eqn9" WIDTH="100"/>

Together, the number of computations (Cp) can be written as the following (where V is the number of validation images): 

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn11.png" alt="eqn11" WIDTH="300"/>


We note that since  k << d, the Nd^2 term is going to dominate the computation. To see how much our problem can be parallelized, we time how long the parallelizable portion of the code runs, and the code's overhead time. We run on one node, and vary the number of threads from 1 to 8. Thus, the time to run on a single node can be written as the following:

<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/eqn12.png" alt="eqn12" WIDTH="300"/>

The results of running on several cores for 40,000 images are shown below in Figure 4. As more threads are added, the parallel component of the algorithm speeds up until reaching 8 cores, at which it stabalizes. The overhead component begins to slightly increase as threads are added, indicating the increased communication that comes with adding more processors. These results show up that for a problem size of 40,000 images, we approach maximum OpenMP parallelization around 8 cores.

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/amdahl.png" alt="amdahl" WIDTH="500"/>
<figcaption> Figure 4: Parallel speedup on a single node varying threads, broken in overhead and parallelizable components on the MNIST dataset. </figcaption>
</figure>
                                                                                                                            

**Hybrid OpenMP + MPI - Model Parallelism**

Using the model parallel framework described above (OpenMP on matrix multiplications, MPI on lambdas), we achieve the following results (Figure 5) when varying threads and nodes ([Driver for hybrid model parallelism - MNIST](https://github.com/dcusworth/image_spark_mpi/blob/master/model/HYBRID/benchmark_model_parallel.py), [OpenMP model parallelism - MNIST](https://github.com/dcusworth/image_spark_mpi/blob/master/model/HYBRID/train_openmp_model.pyx)). We see the maximum speedup occuring with the maximum number of nodes and threads (8 each). The efficiency drops as we increase the number of threads and nodes, but the scaled speedup is still largest for the highest number of threads and nodes. We note that when benchmarking the serial code (hybrid code with 1 node, 1 thread), the algorithm does not scale directly as a function of increased number of images. The scaling is more of the order of Time(serial) = 0.004 N^1.004 + 5. In other words, the overhead does not scale with number of trained images. For some number of nodes, the MPI efficiency is limited as we parallelize over 10 lambdas, some nodes may idle near the end when the number of nodes is not a factor of the number of lambdas.
 

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/model_hybrid.png" alt="model_par" WIDTH="900"/>
<figcaption> Figure 5: Speedup, Scaled Speedup, and Efficiency for hybrid model parallelism on the MNIST dataset. </figcaption>
</figure>
<br />   
                                                                                                                            

We also ran model parallelism on the dataset of our own images - pictures of hands labelled from 0-5 ([Driver for hybrid model parallelism - own images](https://github.com/dcusworth/image_spark_mpi/blob/master/model/HYBRID/benchmark_model_parallel_fingers.py), [OpenMP model parallelism - own images](https://github.com/dcusworth/image_spark_mpi/blob/master/model/HYBRID/train_openmp_model_own.pyx)). We trained the full set of our own images (16,000 images) serially in 470 seconds, and achieved a prediction accuracy of 87% on the reserved validation set. The MNIST dataset by comparison took 70 seconds to serially train 16,000 images. This is due to the higher pixel resolution of our own images (60x40) compared to the MNIST dataset (28x28). We benchmark our algorithm for 3,000 images varying both nodes and threads between 1 and 8. We see a maximum speedup of 6 by using 8 nodes with 8 cores. Compared to the MNIST dataset, we see better speedup and efficiency. This is caused by the higher pixel resolution of the images, leading to a larger speedup related to the tiling parallel matrix multiplication. 

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/own_hybrid.png" alt="model_par" WIDTH="700"/>
<figcaption> Figure 6: Speedup and Efficiency for hybrid model parallelism on our own dataset. </figcaption>
</figure>
<br />  


**Hybrid OpenMP + MPI - Data Parallelism**

Using the data parallel framework described above (OpenMP on matrix multiplications, MPI on subsets of the images), we achieve the following results (Figure 7) when varying threads and nodes([Driver for hybrid data parallelism](https://github.com/dcusworth/image_spark_mpi/blob/master/model/HYBRID/benchmark_data_parallel.py), [OpenMP for data parallelism](https://github.com/dcusworth/image_spark_mpi/blob/master/model/HYBRID/train_openmp_data.pyx)). Similar to the model parallel framework, we see maximum speeup for the maximum number of threads and nodes. However, the speedups are much larger in the data parallel framework than the model parallel framework (25x versus 4x, respectively). We also see much better efficiency in the data parallel approach, where the efficiency remains near optimal for many thread, node configurations.

<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/data_hybrid.png" alt="data_par" WIDTH="900"/>
<figcaption> Figure 7: Speedup, Scaled Speedup, and Efficiency for hybrid data parallelism on the MNIST dataset. </figcaption>
</figure>
<br />   
                                                                                                                            

**Spark parallelization**

Figure 8 shows the results for both outer and inner parallelism ([Spark-outer](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_spark_outer.py)) ([Spark-inner](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_spark_inner.py)) ([AWS serial implementation](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS/aws_serial.py)). We see around 7x speedup for the outer loop Spark implementation. The inner loop implementation runs nearly the same as the serial code. We hypothesize that this is due to the fact that the MNIST dataset's pixel dimension is low, meaning that the parallelization from just inner-most matrix multiplication provides little speedup over the serial version. However, the outer-loop speedup fits between the model and data parallel results of MPI+OpenMP. 

 
<figure>
<img src="https://github.com/dcusworth/image_spark_mpi/blob/master/img/spark_speedup.png" alt="spark" WIDTH="450"/>
<figcaption> Figure 8: Speedup from Spark parallelism on the MNIST dataset run on AWS clusters. </figcaption>
</figure>
<br />  
                                                                                                                            

We were only able to run Spark for 20,000 images in the MNIST dataset, as the outer loop Spark code ran out of memory. 


### Project Deliverables

Deliverable        | Our approach                | Link
-------------------|-----------------------------|--------------
Architecture       | Odyssey - 8 nodes, 8 threads; AWS - 1 master, 4 workers | See report
Hybrid Parallelism | OpenMP + MPI                | [Model + Data](https://github.com/dcusworth/image_spark_mpi/blob/master/model/HYBRID)  
Advanced Feature   | Spark - AWS Cluster , Own image dataset        | [Outer + Inner](https://github.com/dcusworth/image_spark_mpi/blob/master/model/AWS)
Weak/Strong Scaling| Figures 4 - 8             | See report
Computation Graph  | Model & Data DAG            | [Model](https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_1.png), [Data](https://github.com/dcusworth/image_spark_mpi/blob/master/img/dag_2.png)



### Conclusions
We find that the hybrid data parallel OpenMP + MPI gives us the best performace on the image classification problem. Spark gives performance between that of the model and data parallel hybrid algorithms.

The greatest computational bottleneck is in the computation of the pseudo-inverse. We implement a tiled matrix multiplication algorithm in OpenMP to reduce the time it takes to compute the X^T * X before finding the inverse. Future work can be done to implement a hybrid (MPI + OpenMP) inverse algorithm, which will greatly improve performace for images that have a larger number of pixels than the MNIST dataset.

Using a simple regularized linear classifier, we obtain good predictive ability on the MNIST dataset. If the future, if we were to use more complicated images (e.g., classifying faces, expressions, etc.), we would need to implement a more sophisticated classifier, similar to those in use by industry (e.g., AlexNet, GoogleNet, etc.). However, such implementations lose the ability to solve analytically for a solution, and rely on optimization techniques like gradient descent.
