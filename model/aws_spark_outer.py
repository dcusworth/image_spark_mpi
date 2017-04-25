import numpy as np
import time
import random

import pyspark
sc = pyspark.SparkContext()


########### MAKE DATA SELECTION ###########
which_data = 'MNIST'
###########################################

###Choosing to use MNIST dataset
if which_data == 'MNIST':

    import mnist
    images = mnist.train_images()
    labels = mnist.train_labels()

    d = 784 #Pixels of MNIST data

###Choosing to use our own images
elif which_data == 'OWN':


    #Reading own images
    import matplotlib.image as mpimg
    import glob

    #Initialize lists
    images_in = []
    labels_in = []

    #Read 6 classes
    for fingers in np.arange(6):
        list_images = glob.glob("Own_Data/class_"+str(fingers)+"/*.png")
        
        for i in np.arange(len(list_images)):
            #Read image
            single_image = mpimg.imread(list_images[i])[:,:,0] #Select only layer 1
            
            #Make full binary
            single_image[single_image<0.5] = 0
            single_image[single_image>0.5] = 1
            
            #Reshape to 1D
            single_image = list(np.reshape(single_image,[120*180]))
            
            #Add image to the list
            images_in.append(single_image)
            
            #Add label to the list
            labels_in.append(fingers)

    #Shuffle (cause we're reading in order, that messes up selection below)
    #We can remove this if we're using the entire database

    #Initialize lists
    images = []
    labels = []

    #Shuffle
    index_shuf = range(len(labels_in))
    random.shuffle(index_shuf)
    for i in index_shuf:
        images.append(images_in[i])
        labels.append(labels_in[i])

    print('Images: ', np.shape(images))
    print('Labels: ', np.shape(labels))

    d = 21600 #Pixels of finger data


#Labeler function
def label_func(x, choose_label):
    if x == choose_label:
        return 1
    else:
        return -1

#Iterate over different sizes of the training set
for N in range(1000, 60000, 10000):

    start = time.time()

    #Retrieve data and labels - do preprocessing
    y_labs = labels[0:N]

    #Loop over set of regularization parameters
    vaccs = []
    lambdas = [10**q for q in np.linspace(-5,5,10)]
    label_dat = range(10)

    #Load images
    feature_map = np.zeros((N,d))
    for i in range(N): #Just do a subset of training for now
        feature_map[i,:] = images[i].reshape(d)

    #Start spark instance on points
    #Take train test split
    sinds = range(N)
    random.shuffle(sinds)
    tint = int(.8*N)
    tind = sinds[0:tint]
    vind = sinds[tint:-1]

    #Center - i.e. remove mean image
    fpoints = sc.parallelize(feature_map)
    fmean = fpoints.map(lambda x: x).reduce(lambda x,y: (x+y) ) / float(N)
    x_c = fpoints.map(lambda x: x-fmean).collect()

    #Create Spark context for feature matrix
    x_t = sc.parallelize(list(enumerate(x_c))).filter(lambda x: x[0] in tind).map(lambda x: x[1])
    xtb = sc.broadcast(x_t.collect())
    x_v = sc.parallelize(list(enumerate(x_c))).filter(lambda x: x[0] in vind).map(lambda x: x[1])
    xvb = sc.broadcast(x_v.collect())

    #Get training/test labels
    ytrain = sc.parallelize(list(enumerate(y_labs))).filter(lambda x: x[0] in tind).map(lambda x: x[1]).collect()
    y_val = sc.parallelize(list(enumerate(y_labs))).filter(lambda x: x[0] in vind).map(lambda x: x[1]).collect()
    tpoints = sc.parallelize(zip(ytrain, xtb.value))


    #Get pseudo-inverse
    pseudo_inv = sc.parallelize(zip(lambdas, [np.asarray(xtb.value)] * len(lambdas)))\
        .map(lambda x: (x[0], np.linalg.inv(np.dot(x[1].T, x[1]) + np.eye(x[1].shape[1])*N*x[0])))\
        .collect()

    #Get transformation of Y - i.e. X^T * Y
    XtY= sc.parallelize(zip(label_dat, [np.asarray(ytrain)] * len(label_dat), [np.asarray(xtb.value)] * len(label_dat)))\
        .map(lambda x: (x[0], [label_func(q, x[0]) for q in x[1]], x[2]))\
        .map(lambda x: (x[0], np.dot(x[2].T, x[1])))\
        .collect()

    #Get combinations of labels and lambdas
    def solution_func(ipseudo, XtY):

        iouts = sc.parallelize(XtY)\
            .map(lambda x: (x[0], np.dot(ipseudo[1], x[1])))\
            .map(lambda x: (x[0], np.dot(np.asarray(xvb.value), x[1])))\
            .sortByKey(True)\
            .map(lambda x: x[1])\
            .collect()

        out_pred = zip(*iouts)
        ipred = sc.parallelize(zip(*iouts)).map(lambda x: np.argmax(x)).collect()
        
        return (ipseudo[0], ipred)

    #Run over all lambdas
    sols = []
    for ipinv in pseudo_inv:
        sols.append(solution_func(ipinv, XtY))

    #Find the best lambda
    best_sol = sc.parallelize(sols)\
        .map(lambda x: (x[0], np.sum([y == p for y,p in zip(y_val, x[1])]) / float(len(x[1]))))\
        .max(lambda x:x[1])
        
    end = time.time()

    with open('spark_outer.txt', 'a') as myfile:
        myfile.write('validation accuracy = ' + str(best_sol[1]))
        myfile.write('best lambda = ' + str(best_sol[0]))
        myfile.write('elapsed time for ' + str(N) + ' samples = ' + str(end-start))





