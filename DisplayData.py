import math 
import numpy as np
import matplotlib.pyplot as plt
import random

def list2mat(inList,row,col):
	if row*col != len(inList):
		re =  -1
	else:
		re = [[inList[col*i+j] for i in range(col)] for j in range(row)]
	return re


def implot(inList):
	ll = int(math.sqrt(len(inList)));
	image = list2mat(inList,ll,ll);
	fig = plt.figure()
	plt.imshow(image, cmap = plt.get_cmap('gray'), vmin = 0, vmax = max(inList))
	return fig
	
def stackImages(inList,nRows,nCols,ll):
	#reshape each line
	ii = [list2mat(inList[i],ll,ll) for i in range(len(inList))] ## array of matrices
	#append with zeros
	for k in range(len(ii),nRows*nCols):
		ii.append(np.zeros((ll,ll)))
	#stack all matrices
	jj= np.vstack([np.hstack(ii[i*nCols:(i+1)*nCols]) for i in range(nRows)])
	return jj
	
def displayData(X,nRows,nCols):
	# check that input is a list
	if type(X) is not list:
		X = np.ndarray.tolist(X)
		
	nImages = len(X)
	imageSize = int(math.sqrt(len(X[0])));
	if type(nRows) is not int :
		#calculate rows & collums
		nRows = int(math.floor(math.sqrt(nImages)));
		nCols = int(math.ceil(float(nImages) / nRows));
	image = stackImages(X,nRows,nCols,imageSize)
	fig = plt.figure()
	plt.imshow(image, cmap = plt.get_cmap('gray'), vmin = min(min(X)), vmax = max(max(X)))
	return fig
		

def displayPca1(X,U,inds):
		#display PCA gradients of selectes sumples
		if(len(inds)==0):
			inds = random.sample(range(len(X)),10);
		samples = X[[inds]]
		pc = (samples*U)
		restore = pc*np.transpose(U)
		nSamples = len(pc)
		allMat = np.matrix([])
		nEvDisp = 10
		for k in range(nSamples):
			#find largest valuse
			sortIndex = np.argsort(-np.abs(pc[k]))[:,0:nEvDisp]
			a =  np.vstack([U[:,sortIndex[0,i]]*pc[k,sortIndex[0,i]] for i in range(nEvDisp)])
			# restore image
			b = np.sum(a,0)
			#append them
			if(len(allMat)==1):
				#start loop
				allMat = np.vstack([samples[k], a, b,restore[k] ])
			else:
				allMat = np.vstack([allMat, samples[k], a,b,restore[k]])
		f = displayData(allMat,nSamples,nEvDisp+3)
		f.suptitle('PCA')
		return f			