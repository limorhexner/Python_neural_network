#all learning algorithm functions
import numpy as np

def costFunction(nnParams,inMat,outVec,hiddenLayerSize,cLambda):
	#calculate cost function and gradient in corrent parameters
	nSamples, inSize = inMat.shape;
	outSize  = len(np.unique(outVec.tolist()));
	#reshape parameters into matrixes
	l1 = hiddenLayerSize*(inSize+1);
	l2 = (hiddenLayerSize+1)*outSize;
	if (len(nnParams)!=l1+l2):
		print 'wrong length of nnParams'		
		return -1
	theta1 = nnParams[0:l1].reshape(hiddenLayerSize,inSize+1);
	theta2 = nnParams[l1:(l1+l2)].reshape(outSize,hiddenLayerSize+1);
	
	# Part 1: Feedforward the neural network and return the cost
	a1 = np.hstack((np.ones((nSamples,1)),inMat)) #add bias parameter
	z2 = a1*theta1.transpose();
	a2 = np.hstack((np.ones((nSamples,1)),sigmoid(z2)));
	z3 = a2*theta2.transpose();
	a3 =  sigmoid(z3);
	
	hTheta =a3;
	yMat = np.zeros((nSamples,outSize));
	tmp = [yMat.itemset((i,outVec[i]-1),1) for i in range(nSamples)];
	tmp = -np.multiply(yMat,np.log(hTheta));
	tmp -= np.multiply(1-yMat,np.log(1-hTheta));
	cost = np.sum(tmp);
	
	J = cost/nSamples + cLambda/(2*m)*(np.sum(np.power(theta1,2)) + np.sum(np.power(theta2,2)))
	return J


	
def sigmoid(z):
	#calculase sigmoid fanction over a vector
	return 1.0 / (1.0 + np.exp(-z));

