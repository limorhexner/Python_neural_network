import numpy as np
from costFunc import costFunction
import random

def gradientDescent(inMat,outVec,initParams,hiddenLayerSize,nIteration,validationSize):
	#======================================
	#run gradient descent algorithm 
	#to find neural network parameters
	#input: 
	#   inMat: input matrix (nSample X inSize)
	#   outVec: samples lable (nSample X 1)
	#   hiddenLayerSize- network parameter
	#   nIteration: num of iterations to run
	#   validationSize: size of velidation set, used for debug
	#output:
	#   nnParams: neural network parameters
	#   labels: labels in training set
	#   Jhistory,JhistoryVal: cost Vs iteration of trainig ans validation set, used for debug
	#=========================================
	
	#constant values
	cLambda = 0.1
	stepSize = 10
	#data parameters
	nSamples, inSize = inMat.shape
	nTrainSamples = nSamples-validationSize
	outSize  = len(np.unique(outVec.tolist()))
	labels = np.unique(outVec.tolist())
	#initiate output
	Jhistory = np.zeros(nIteration);
	JhistoryVal = np.zeros(nIteration)
	#cut trainig set
	if validationSize>0:
		inds = np.random.permutation(nSamples)
		Xt = inMat[inds[:nTrainSamples],:]
		yt = outVec[inds[:nTrainSamples]]
		Xv = inMat[inds[nTrainSamples:],:]
		yv = outVec[inds[nTrainSamples:]]
	else:
		Xt = inMat
		yt = outVec
	#run iterations
	nnParams = initParams
	J,grad = costFunction(nnParams,Xt,yt,labels,hiddenLayerSize,cLambda)
	for ii in range(nIteration):
		#start with defaoult steo size
		tStep = stepSize
		while True:
			#go one step and calculate cost
			newParams = nnParams - tStep*grad
			Jn,gradNew = costFunction(newParams,Xt,yt,labels,hiddenLayerSize,cLambda)
			if Jn<J: #cost decreased- continue next step
				nnParams = newParams
				J=Jn
				grad = gradNew
				break
			else: #take smaller step
				tStep /= 5
		Jhistory[ii] = J
		if validationSize>0:
			# cost of validation set
			Jt,tmp = costFunction(newParams,Xv,yv,labels,hiddenLayerSize,cLambda)
			JhistoryVal[ii] = Jt
		if np.mod(ii,20)==0:
			print 'iteration ',ii,' cost: ', J
	return nnParams, labels, Jhistory,JhistoryVal
			