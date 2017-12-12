import numpy as np
import matplotlib.pyplot as plt

def load_data():
	data = np.load('clean2.npy')
	return data

def plot_histogram(data, xlabel, ylabel):
	n, bins, patches = plt.hist(data,bins='fd',range=(0,2000000))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid(True)
	plt.show()
	#plt.savefig(xlabel+'-'+ylabel+'-Histogram',format='svg')

def plot_scatter(x, y, xlabel, ylabel):
	plt.scatter(x,y,marker='+')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()
	#plt.savefig(xlabel+'-'+ylabel+'-ScatterPlot',format='svg')

data = load_data()
#plot_histogram(data[:,2],'Sold Prices','Frequency')

a = np.fabs(data[:,2] - data[:,3])
print(np.mean(a))
b = np.divide(a, data[:,2])
print('Percentage of data where list price is within 3% of sold price', np.mean(b <= 0.03))
print('Percentage of data where list price is within 10% of sold price', np.mean(b <= 0.1))
print('Percentage of data where list price is within 20% of sold price', np.mean(b <= 0.20))
print(100*np.mean(b))
#plot_scatter(data[:,3], data[:,2], 'List Price', 'Sold Price')
#plot_scatter(data[:,6], data[:,2], 'Square footage', 'Sold Price')
