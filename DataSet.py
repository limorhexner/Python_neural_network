import DisplayData as dd
import numpy as np
import random

class dataSet:

	def __init__(self):
		#initiate dataset
		self.X=[]
		self.y=[]
		self.pcaExist = False
		self.pEn = 99.0
		self.nSamplesPlot = 20
		self.nEvPlot = 15
		
	def readInput(self,inFile):
		self.X = readCsv(inFile)
		
	def readOutput(self,inFile):
		self.y = readCsv(inFile).transpose();

	def displayInput(self,nDisp,isRand):
		#display nDisp samples from input in a grid.
		X = self.X
		if isRand:
			XX = random.sample(np.matrix.tolist(X),nDisp)
		else:
			XX = X[0:nDisp]
		f = dd.displayData(XX,[],[])
		f.suptitle('input samples')
		self.inFig = f
		
		
	def displayPca(self,inds):
		#display PCA largest ingradients of selectes sumples
		X = self.X
		U = self.U
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
		f = dd.displayData(allMat,nSamples,nEvDisp+3)
		f.suptitle('PCA')
		self.pcaFig = f		
			
		
	def runPca(self):
		if len(self.X)==0:
			print 'dataSet object have no data, use ds.readInput'
			return
		#make sur pEn is float
		if type(self.pEn) is not float:	
			self.pEn = float(self.pEn)
		self.U, self.nd = PCA(self.X,self.pEn)
		self.Z= self.X*self.U
		self.pcaExist = True
		
	
def PCA(inMat,pEn):
	#init outMat
	m,n=np.shape(inMat)
	covMat = (np.zeros((n,n)))
	# calc covMat
	for i in range(m):
		x=inMat[i]
		covMat += np.dot(np.transpose(x),x)/m
	# get SVD
	U,S,v =np.linalg.svd(covMat);
	s = np.cumsum(S)/np.sum(S);
	nd = len([x for x in s if x<pEn/100])+1
	outMat = U[0:n,range(nd)]	
	return outMat, nd

def readCsv(name):
	import csv
	res =[]; 
	csvReader = csv.reader(open(name, 'r'), delimiter=',',quotechar='|');
	for row in csvReader:
		frow = [float(row[i]) for i in range(len(row))];
		res.append(frow);
	#chec if vector or matrix
	if len(res)==1:
		return np.matrix(frow)
	else:
		return np.matrix(res)
	