import numpy as np
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt


rcParams['figure.figsize'] = 14, 8

def mse(y_true, y_pred):
    error = (y_pred-y_true) ** 2
    return np.mean(error)

def plot_mse(num=100):
    X = np.linspace(start=-100, stop=100, num=num)
    y = np.square(X)
    plt.plot(X, y)
    plt.xlabel("y_pred")
    plt.ylabel("loss")
    plt.title("Plot")
    plt.show()