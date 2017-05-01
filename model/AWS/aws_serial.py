import numpy as np
import time
import random

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

for N in range(1000, 60000, 10000):

    start = time.time()

    #Retrieve data and labels - do preprocessing
    y_labs = labels[0:N]

    #Loop over set of regularization parameters
    vaccs = []
    lambdas = [10**q for q in np.linspace(-5,5,10)]

    #Load images
    feature_map = np.zeros((N,d))
    for i in range(N): #Just do a subset of training for now
        feature_map[i,:] = images[i].reshape(d)

    #Start spark instance on points
    #Take train test split
    sinds = range(N)
    random.shuffle(list(sinds))
    tint = int(.8*N)
    tind = sinds[0:tint]
    vind = sinds[tint:-1]

    #Get rid of bias
    fmean = feature_map.mean(axis=0)
    x_c = feature_map - fmean[np.newaxis,:]

    Xtr = x_c[0:tint]
    Xvl = x_c[tint:-1]
    y_val = y_labs[tint:-1]
    y_tr = y_labs[0:tint]
    
    Nt = Xtr.shape[0]
    Mt = Xtr.shape[1]

    for ll in lambdas:

            ws = []
            iouts = []
            classes = []
                
            #Get denominator - depends on lambda/regularization and not label
            denom_sum = np.linalg.inv(np.dot(X.T, X) + np.eye(d)*ll)

            #Loop over all labels
            for choose_label in range(6): 

                y_tr_map = [label_func(q, choose_label) for q in y_tr]

                numer_sum = np.zeros(Mt)
                for i in range(Nt):
                    x_iT = Xtr[i,:]
                    inumer = x_iT * y_tr_map[i]
                    numer_sum += inumer

                iw = np.zeros(Mt)
                for i in range(Mt):
                    for j in range(Mt):
                        iw[i] += denom_sum[i][j] * numer_sum[j]


                iout = np.zeros(V)
                for i in range(V):
                    for j in range(Mt):
                        iout[i] += Xvl[i][j] * iw[j]

                iclass = [np.sign(q) for q in iout]

                #Append to output
                ws.append(iw)
                iouts.append(iout)
                classes.append(iclass)

            #Figure out how to spark-ify this loop
            out_pred = list(zip(*iouts))

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
    with open('serial_' + which_data + '.txt', 'a') as myfile:
        myfile.write('validation accuracy = ' + str(vaccs[best_val]) + '\n')
        myfile.write('best lambda = ' + str(lambdas[best_val]) + '\n')
        myfile.write('elapsed time for ' + str(N) + ' samples = ' + str(end-start) + '\n')

