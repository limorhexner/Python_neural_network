import numpy as np
import random
from DataSet import dataSet, readCsv
from gradDescent import gradientDescent

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
		self.nIteration = 40;
	
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
			self.U = ds.U; #save for future validation
		else:
			inSet = ds.X;
		
		self.theta,self.labels,self.inSize = nnTrain(inSet,ds.y,self.hiddenLayerSize,self.nIteration)


	def predict(self,X):
		#validate size of theta
		#validate PCA match
		#if not applied- apply PCA
		if(len(self.theta)==0):
			print 'neural network not initialised'
			return -1
		
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
	
