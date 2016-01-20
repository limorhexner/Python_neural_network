import numpy as np

def costFunction(nnParams,inMat,outVec,lables,hiddenLayerSize,cLambda):
	#===========================================================
	#calculate cost function and gradient for input parameters
	#input: 
	#   nnParams: nueral network parameters unrolled
	#   inMat: input matrix (nSample X inSize)
	#   outVec: samples lable (nSample X 1)
	#   hiddenLayerSize- network parameter
	#   cLambda: regularization parameter. use cLambda = 0 for unregulized network
	#output:
	#  J: cost for input parameters(scalar)
	#  grad: parameters gradient unrolled
	#===========================================================
	
	nSamples, inSize = inMat.shape
	outSize  = len(lables)
	#input size validation
	if len(outVec)!= nSamples:
		print 'wrong output length'		
		return -1 ,-1,-1,-1
		
	theta1,theta2,l1,l2 = arrangeParams(nnParams,inSize,outSize,hiddenLayerSize)	
		
	#===Part 1: Feedforward the neural network and return the cost===
	#add bias parameter
	a1 = np.hstack((np.ones((nSamples,1)),inMat)) 
	#calculate neuron layer (nSamples X hiddenLayerSize)
	z2 = a1*theta1
	#add bias neuron
	a2 = np.hstack((np.ones((nSamples,1)),sigmoid(z2)))
	#calc output layer (nSamples X outSize)
	z3 = a2*theta2
	a3 = sigmoid(z3)
	hTheta =a3;
	
	#calculate cost
	#true out matrix
	yMat = np.zeros((nSamples,outSize));
	tmp = [yMat.itemset((i,outVec[i]),1) for i in range(nSamples)];#(tmp is to supress print)
	#cost = -y*log(hTheta) - (1-y)*log(1-hTeta)
	tmp = -np.multiply(yMat,np.log(hTheta)) #first part
	tmp -= np.multiply(1-yMat,np.log(1-hTheta))#second part
	J = np.sum(tmp)/nSamples #average
	#add regularization over the parameters cost += sum(theta^2)
	J+= cLambda/(2*nSamples)*np.sum(np.power(theta1[1:inSize+1,:],2))
	J+= cLambda/(2*nSamples)*np.sum(np.power(theta2[1:hiddenLayerSize+1,:],2))
	
	#===Part 2: Feedbackward the neural network and return the gradient===
	delta3 = (a3-yMat).transpose() #(outSize X nSamples)
	gTag2 = sigmoidGradient(z2.transpose()) #(hiddenLayerSize X nSamples)
	delta2 = (theta2*delta3); #(hiddenLayerSize+1 X nSamples)
	delta2 = np.multiply(delta2[1:hiddenLayerSize+1,:],gTag2) #(hiddenLayerSize X nSamples)

	# accumulate delta
	Delta1 = np.matrix(np.zeros((hiddenLayerSize,inSize+1)));
	Delta2 = np.matrix(np.zeros((outSize,hiddenLayerSize+1)));
	for k in range(nSamples):
		Delta1 += delta2[:,k]*a1[k,:];
		Delta2 += delta3[:,k]*a2[k,:];
	theta1Grad = Delta1.transpose()/nSamples;
	theta2Grad = Delta2.transpose()/nSamples;
	#add regularization 
	theta1Grad[1:inSize+1,:]+= cLambda/nSamples*theta1[1:inSize+1,:]
	theta2Grad[1:hiddenLayerSize+1,:]+= cLambda/nSamples*theta2[1:hiddenLayerSize+1,:]
	#unroll gradient
	grad = np.vstack((theta1Grad.reshape((l1,1)),theta2Grad.reshape((l2,1))))

	return J, grad


#used functions
def sigmoid(z):
	#calculase sigmoid fanction over a vector
	return 1.0 / (1.0 + np.exp(-z));

def sigmoidGradient(z):
	#the gradient of the sigmoid function
	sigz=sigmoid(z);
	return np.multiply(sigz,(1-sigz));
	
def arrangeParams(nnParams,inSize,outSize,hiddenLayerSize):
	#reshape parameters into matrixes
	l1 = hiddenLayerSize*(inSize+1)
	l2 = (hiddenLayerSize+1)*outSize
	if (len(nnParams)!=l1+l2):
		print 'wrong length of nnParams'
		print 'l1=',l1,' l2 =',l2
		print 'nnParams length=' ,len(nnParams)
		return -1 ,-1,-1,-1
	theta1 = nnParams[0:l1].reshape(inSize+1,hiddenLayerSize)
	theta2 = nnParams[l1:(l1+l2)].reshape(hiddenLayerSize+1,outSize)
	return theta1,theta2,l1,l2