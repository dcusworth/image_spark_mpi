# Highly Parallelized Image Recognition via TensorFlow

TensorFlow is a state-of-the-art open source library for machine learning developed by Google. The Python library (using GCC) allows machine learning using neural networks, a method very suited for image recognition. Unlike basic machine learning, neural networks use layers that are shaped based on the problem studied. A classic example is the recognition of written numbers based on a model calibrated by thousands of written numbers. This type of image recognition is also applied in self-driving vehicles that have to identify different objects on the road and respond based on their location and movement.

Here, we will use TensorFlow to set up a system that can interpret photos of hands holding up a number of fingers. The model should be able to (in close to real-time) calculate the sum of the numbers indicated by the fingers raised by two hands. 

TensorFlow has the capability to be run accelerated on GPUs and since March 2016, it has internal options to run distributed. The distributed version of TensorFlow is supported by gRPC and allows different parts of the graph to be computed in parallel on different cores (which can be part of different nodes). We will use these options to optimize the algorithm as much as possible. Some experimental work has been done on running TensorFlow with MPI [https://arxiv.org/pdf/1603.02339]. An attempt will be made to use MPI with Tensorflow as well. If MPI does not work with TensorFlow, using the Theano-MPI framework for deep learning [https://arxiv.org/pdf/1605.08325.pdf] will be considered as an alternative for the MPI component of the project. 

A number of task have been defined to streamline the project:
- Get TensorFlow to work on Odyssey.
- Get The Distributed version of TensorFlow to work on Odyssey GPUs.
- Implement our own set of photos. This includes parallelized pre-processing which will prepare the photos bto be used as input for the model.
- Attempt to parallelize with MPI. Study Theano-MPI if this proofs unviable. 
- Run a parallel version of TensorFlow with our own photos.
- Benchmark the GPU implementation of TensorFlow with our own dataset with different architecture parameters. Analyze speedup, efficiency, iso-efficiency. Compare a version with and without MPI. Evaluate performance, overhead, and scheduling. 
- Set up a way in which new photos can be analyzed and interpreted quickly.

The project will presented in the form of a report, website, and presentation. 
