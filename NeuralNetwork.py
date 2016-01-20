import numpy as np
import random
from DataSet import dataSet, readCsv
from gradDescent import gradientDescent
import costFunc
import DisplayData as dd

class neuralNetwork:
#this class is a neural network representation
#methods: 
#  learnNetwork
#  testNetwork
#  predict
#parameters:
#  network parameters (available after learning or update)
	def __init__(self,hiddenLayerSize):
		#initiate network- no weights
		theta  = [];
		self.hiddenLayerSize = hiddenLayerSize;
		self.nIteration = 150;
	
	def setNetworkParameters(self,thetaVec):
		#manually set network parameters
		#does not validate size
		self.theta = thetaVec;
		
	def readNetworkParams(self,csvFile):
		#read network parameters from csv file
		self.theta = readCsv(csvFile)
	
	def learnNetwork(self,ds):
		#learn network parameters from training data set
		#input: ds class DataSet
		if(len(ds.y)==0):
			print 'dataSet object have no output data, use ds.readOutput'
			return -1
		#get parameters
		self.usePca = ds.pcaExist #learn over PCA
		if self.usePca:
			inSet = ds.Z;
			self.U = ds.U; #save for future run
			self.isPCA = True
		else:
			inSet = ds.X;
			self.isPCA = False
		self.theta,self.labels,self.inSize = nnTrain(inSet,ds.y,self.hiddenLayerSize,self.nIteration)
		self.outSize = len(self.labels)

	def predict(self,X):
		#use neural network to estimate y
		if(len(self.theta)==0):
			print 'neural network not initialised'
			return -1
		#X can be matrix or dataSet
		if type(X) is np.matrixlib.defmatrix.matrix:
			inMat = X
		else:
			#input is dataset
			if self.isPCA:
				#wrok on pca
				inMat = X.X*self.U
			else:
				#work on full input
				inMat = X.X
		
		return feedForward(self.theta,inMat,self.hiddenLayerSize,self.outSize)

	def nnTest(self,ds):
		#test network parameters with labeled dataSet
		p = self.predict(ds)
		y = ds.y
		inds = (p==y)
		print 'Success rate: ' ,float(inds.sum())/float(inds.size)*100,'%'
		#display correct sumples
		tInd = np.where(p==y)[0].tolist()[0]
		if len(tInd)>50:
			#randomly choose 50
			tInd = random.sample(tInd,50)
		dispMat = ds.X[tInd]
		f = dd.displayData(dispMat,[],[])
		f.suptitle('correct samples')
		self.correctFig = f
		#display error sumples
		errInd = np.where(p!=y)[0].tolist()[0]
		if len(errInd)>50:
			#randomly choose 50
			errInd = random.sample(errInd,50)
		dispMat = ds.X[errInd]
		f = dd.displayData(dispMat,[],[])
		f.suptitle('error samples')
		self.errFig = f

		
		
		
#private functions
def nnTrain(inSet,outVec,hiddenLayerSize,nIteration):
	#train nueral network
	nSamples, inSize = inSet.shape;
	outSize=len(np.unique(np.matrix.tolist(outVec)));
	nParameters = (inSize+1)*hiddenLayerSize + (hiddenLayerSize+1)*outSize;
	validationSize = 0;
	#randomly initiate parameters
	initParams = randInitializeWeights(nParameters)
	theta,labels,j1,j2 = gradientDescent(inSet,outVec,initParams,hiddenLayerSize,nIteration,validationSize)
	return theta,labels,inSize

def randInitializeWeights(nParameters):
	#randomly initiate weights matrix mXn
	epsilonInit = 0.12;
	W = np.random.random((nParameters,1))* 2 * epsilonInit - epsilonInit;
	return W
	
def feedForward(nnParams,inMat,hiddenLayerSize,outSize):
	nSamples, inSize = inMat.shape
	theta1,theta2,l1,l2 = costFunc.arrangeParams(nnParams,inSize,outSize,hiddenLayerSize)
	if type(theta1) is int:
		#if error will be -1
		return -1
		
	a1 = np.hstack((np.ones((nSamples,1)),inMat)) 
	z2 = a1*theta1
	a2 = np.hstack((np.ones((nSamples,1)),costFunc.sigmoid(z2)))
	hTheta = costFunc.sigmoid(a2*theta2)
	return np.argmax(hTheta,1)
	
	
