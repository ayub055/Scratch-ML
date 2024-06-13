import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')

    xlim = (min(X[:,0]))-2, max(X[:,0]+2)
    ylim = (min(X[:,1]))-2, max(X[:,0]+2)

    xg = np.linspace(xlim[0], xlim[1], 60)
    yg = np.linspace(ylim[0], ylim[1], 60)

    xx, yy = np.meshgrid(xg, yg)
    Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T


    for label, color in enumerate(['red', 'blue']):
        mask = (y==label)
        mu, std = X[mask].mean(0), X[mask].std(0)
        P = np.exp(-0.5 * (Xgrid-mu) ** 2 / std ** 2).prod(1)
        Pm = np.ma.masked_array(P, P<0.03)
        ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5, cmap=color.title()+'s')
        ax.contour(xx, yy, P.reshape(xx.shape), levels=[0.01, 0.1, 0.5, 0.9],
                colors = color, alpha=0.2)
    ax.set(xlim=xlim, ylim=ylim)
    plt.title('Gaussian Distributed Data')
    plt.show()
