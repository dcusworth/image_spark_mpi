import numpy as np
import time
import random

from mnist import MNIST


def label_func(x, choose_label):
	if x == choose_label:
		return 1
	else:
		return -1

#Retrieve data and labels - do preprocessing
mndata = MNIST('/Users/thclements/python-mnist/data')
images, labels = mndata.load_training()
images, labels = np.array(images), np.array(labels)
vaccs = []
lambdas = np.logspace(-5,5,num=10)

#Take train test split
N = images.shape[0]
tint = int(.8*N)

#Get rid of bias
fmean = images.mean(axis=0)
x_c = images - fmean[np.newaxis,:]

Xtr = x_c[0:tint,:]
Xvl = x_c[tint:-1,:]
y_val = labels[tint:-1]
y_tr = labels[0:tint]

Nt = Xtr.shape[0]
Mt = Xtr.shape[1]

start = time.time()
xTx = np.dot(Xtr.T,Xtr)

for ll in lambdas:
	ws = []
	iouts = []
	classes = []
	xTx = xTx + np.eye(Mt) * ll
	denom_sum = np.linalg.inv(xTx)

	# Loop over all labels
	for choose_label in range(10): 
		y_tr_map = np.array([label_func(q, choose_label) for q in y_tr])
		numer_sum = np.dot(Xtr.T,y_tr_map)
		iw = np.dot(denom_sum,numer_sum)
		iout = np.dot(Xvl, iw)
		iclass = map(np.sign,iout)

		#Append to output
		ws.append(iw)
		iouts.append(iout)
		classes.append(iclass)

	out_pred = np.array(iouts)

	preds = []
	for idx in range(out_pred.shape[1]):
		iclass = np.where(out_pred[:,idx] == np.max(out_pred[:,idx]))[0][0] 
		preds.append(iclass)

	#Determine accuracy on validation
	vacc = np.sum([y == p for y,p in zip(y_val, preds)]) / float(len(preds))

	#Append to lambda
	vaccs.append(vacc)

end = time.time()

best_val = np.where(vaccs == np.max(vaccs))[0][0]
print('validation accuracy = {}'.format(vaccs[best_val]))
print('best lambda = {}'.format(lambdas[best_val]))
print('elapsed time for {} samples = {} seconds'.format(N,end-start))

