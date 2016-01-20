import math 
import numpy as np
import matplotlib.pyplot as plt
import random
#functions used for image display 

def implot(inList):
	#single image display
	#input: list with length inSize
	#image should be square
	ll = int(math.sqrt(len(inList)));
	image = list2mat(inList,ll,ll);
	fig = plt.figure()
	plt.imshow(image, cmap = plt.get_cmap('gray'), vmin = 0, vmax = max(inList))
	return fig
	
def displayData(X,nRows,nCols):
	#multiple images display in a grid
	#input:
	#  X: image matrix or 2D list(nImages X inSize)
	#  nRows, nCols: grid parameters. if empty grid will be square
	
	# check that input is a list
	if type(X) is not list:
		try:
			X = np.ndarray.tolist(X)
		except:
			print 'error in displayData: input should be list or matrix'
			return
		
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
	
# private functions:		
def list2mat(inList,row,col):
	if row*col != len(inList):
		re =  -1
	else:
		re = [[inList[col*i+j] for i in range(col)] for j in range(row)]
	return re


def stackImages(inList,nRows,nCols,ll):
	#reshape each line
	ii = [list2mat(inList[i],ll,ll) for i in range(len(inList))] ## array of matrices
	#append with zeros
	for k in range(len(ii),nRows*nCols):
		ii.append(np.zeros((ll,ll)))
	#stack all matrices
	jj= np.vstack([np.hstack(ii[i*nCols:(i+1)*nCols]) for i in range(nRows)])
	return jj
	
