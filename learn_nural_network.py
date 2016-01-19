import csv
import os
import math 
import random
import numpy as np
import matplotlib.pyplot as plt
from DisplayData import displayData
from DataSet import dataSet, readCsv
import myMath as mm
from  LearnAlg import costFunction

# os.chdir('C:\Users\Limor_2\Documents\GitHub\Python_neural_network')
#import learn_nural_network as lnn
#reload(lnn)
#import sys

def tesetCostFun():
	
	ds = dataSet()
	ds.readInput('input.csv')
	ds.readOutput('learnOutput.csv')
	ds.runPca();
	thetaInint = readCsv('theta.csv')
	
	J ,grad= costFunction(thetaInint,ds.Z,ds.y,25,0.1)
	print J
	return grad
	
	

def run():
	ds = dataSet()
	ds.readInput('input.csv')
	ds.displayInput(21,True)
	ds.runPca()
	
	plt.show()
	
	return ds



