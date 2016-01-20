# run neural network
import matplotlib.pyplot as plt
from DataSet import dataSet
from  NeuralNetwork import neuralNetwork

def main():
	#arrange and display dataSet
	ds = dataSet()
	ds.readInput('input.csv')
	ds.readOutput('learnOutput.csv')
	ds.displayInput(49,True)
	ds.runPca();
	ds.displayPca([])
	# run neural network
	nn = neuralNetwork(25)
	nn.learnNetwork(ds)
	nn.nnTest(ds)
	plt.show()  

if __name__ == "__main__":
    main()

