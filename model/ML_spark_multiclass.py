import numpy as np
import time
import random

import findspark
findspark.init() 

import pyspark
sc = pyspark.SparkContext()

from mnist import MNIST
mndata = MNIST('/Users/dcusworth/Desktop/mnist/MNIST/python-mnist/data')
images, labels = mndata.load_training()

#Build feature map
N = 5000 #How many images I want to load
d = 784 #Pixels of MNIST data

def bayes_rule(x):
    if x > 0:
        return 1
    else:
        return -1
    
#label_func = lambda x,choose_label: [1 if la == choose_label else -1 for la in x]
def label_func(x, choose_label):
    if x == choose_label:
        return 1
    else:
        return -1

#Retrieve data and labels - do preprocessing
y_labs = labels[0:N]

#Loop over set of regularization parameters
vaccs = []
lambdas = [10**q for q in np.linspace(-5,5,10)]


#Load images
feature_map = np.zeros((N,d))
for i in range(N): #Just do a subset of training for now
    feature_map[i,:] = images[i]

#Start spark instance on points
#Take train test split
sinds = range(N)
random.shuffle(sinds)
tint = int(.8*N)
tind = sinds[0:tint]
vind = sinds[tint:-1]

#Center images here
fpoints = sc.parallelize(feature_map)
fmean = fpoints.map(lambda x: x).reduce(lambda x,y: (x+y) ) / float(N)
x_c = fpoints.map(lambda x: x-fmean).collect()

start = time.time()
for ll in lambdas:

    ws = []
    iouts = []
    classes = []


    ### Loop over all labels
    for choose_label in range(10): 

        #Do binary classification for certain label
        y_label = [label_func(q,choose_label) for q in y_labs]
        tpoints = sc.parallelize(zip([yy for idx,yy in enumerate(y_labs) if idx in tind], \
                                     [xx for idx,xx in enumerate(x_c) if idx in tind]))
        vpoints = sc.parallelize(zip([yy for idx,yy in enumerate(y_labs) if idx in vind], \
                                     [xx for idx,xx in enumerate(x_c) if idx in vind]))

        y_val = vpoints.map(lambda x:x[0]).collect()
        #y_val = [yy for idx,yy in enumerate(y_labs)]

        ###### Analytical solution to problem for certain label #######

        #Do numerator first - doesn't require regularization
        numer_map = tpoints.map(lambda x:x[1] * (label_func(x[0],choose_label))) 
        numer_sum = numer_map.reduce(lambda x,y: x+y)

        #Get denominator - depends on lambda
        denom_map = tpoints.map(lambda x: np.dot(x[1], x[1].T) + N*ll) #Need to add regularization - lambda
        denom_sum = denom_map.reduce(lambda x,y: x+y)
        iw = numer_sum / float(denom_sum)

        #Test on validation set
        ires = vpoints.map(lambda x:np.dot(x[1],iw))
        iout = ires.collect()
        iclass = ires.map(lambda x: bayes_rule(x)).collect()

        #Append to output  - Add MPI communication or further spark-ize
        ws.append(iw)
        iouts.append(iout)
        classes.append(iclass)


    #Figure out how to spark-ify this loop
    out_pred = zip(*iouts)

    preds = []
    for idx in range(len(out_pred)):
        ipreds = np.asarray(out_pred[idx])
        iclass = np.where(ipreds == np.max(ipreds))[0][0] 
        preds.append(iclass)

    #Determine accuracy on validation
    vacc = np.sum([y == p for y,p in zip(y_val, preds)]) / float(len(preds))
    
    #Append to lambda
    vaccs.append(vacc)

end = time.time()

best_val = np.where(vaccs == np.max(vaccs))[0][0]
print 'validation accuracy = ', vaccs[best_val]
print 'best lambda =', lambdas[best_val]
print 'elapsed time for', N, 'samples = ', end-start, 'seconds'
