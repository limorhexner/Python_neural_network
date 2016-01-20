import csv
import os
import math 
import random
import numpy as np
import matplotlib.pyplot as plt
from DisplayData import displayData
from DataSet import dataSet, readCsv
import myMath as mm
from  NeuralNetwork import neuralNetwork

# os.chdir('C:\Users\Limor_2\Documents\GitHub\Python_neural_network')
#import learn_nural_network as lnn
#reload(lnn)
#import sys

def testLearn():
	ds = dataSet()
	ds.readInput('input.csv')
	ds.readOutput('learnOutput.csv')
	ds.runPca();
	nn = neuralNetwork(25)
	nn.nIteration = 20;
	nn.learnNetwork(ds)
	nn.nnTest(ds)
	plt.show()
	
