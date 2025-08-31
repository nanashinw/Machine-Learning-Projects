import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    
    X = data[:,:7]
    y = data[:,7]
    return X, y

def sig(z):
    return 1/(1+np.exp(-z))

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == "Extrovert"
    negative = y == "Introvert"

    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
