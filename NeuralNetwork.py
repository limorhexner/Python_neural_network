import numpy as np
import random
from DataSet import dataSet, readCsv

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
	
	def setNetworkParameters(self,thetaVec):
		#manually set network parameters
		#does not validate size
		self.theta = thetaVec;
	def readNetworkParams(self,csvFile)
		#read network parameters from csv file
		self.theta = csvRead(csvFile)
	
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
		nSamples, inSize = inSet.shape;
		outSize=len(np.unique(np.matrix.tolist(ds.y)));
		nParameters = (inSize+1)*self.hiddenLayerSize + (self.hiddenLayerSize+1)*outSize;
		#randomly initiate parameters
		self.theta = randInitializeWeights(nParameters)
		#TODO- get lable vec from np.unique(outVec.tolist())
		

		self.inSize = inSize; #save for future validation

	def predict(self,X):
		#validate size of theta
		#validate PCA match
		#if not applied- apply PCA
		if(len(self.theta)==0):
			print 'neural network not initialised'
			return -1
		


def randInitializeWeights(nParameters):
	#randomly initiate weights matrix mXn
	epsilonInit = 0.12;
	W = np.random.random((nParameters,1))* 2 * epsilonInit - epsilonInit;
	return W
	
def 