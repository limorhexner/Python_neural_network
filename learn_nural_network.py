import csv
import os
import math 
import random
import numpy as np
import matplotlib.pyplot as plt
from DisplayData import displayData
from DataSet import dataSet
import myMath as mm

#os.chdir('C:\\Users\\Limor_2\\Documents\\coursera\\neuron_network')
#import learn_nural_network as lnn
#reload(lnn)
#import sys
# sys.modules[__name__].__dict__.clear()

def run():
	ds = dataSet()
	ds.readInput('input.csv')
	ds.displayInput(21,True)
	ds.runPca()
	
	plt.show()
	
	return



