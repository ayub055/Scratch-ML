import numpy as np
import pandas as pd
from helper import mse
import matplotlib.pyplot as plt
from dataloader import load_data, split_data


class LinearRegression:
    def __init__(self, num_features):
        self.W = np.zeros(num_features)
    
    def gradient_step(self, X, y, lr, regularize=True):
        y_pred = self.predict(X)
        grad_w = ((y_pred-y).T @ X)/len(X)
        if regularize:
            l2 = 0.1
        else: l2 = 0
        
        self.W -= lr * grad_w 
        self.W -= lr * l2 * np.sum(self.W)
        
    
    def fit(self, X, y, num_iters=10, learning_rate=0.01, use_formula=False):
        loss_history = []
        
        for iter in range(num_iters):
            y_pred = self.predict(X)
            loss = mse(y, y_pred)
            loss_history.append(loss)
            # print(f'Epoch {iter + 1}: train loss {loss:.4f}')
            self.gradient_step(X, y, learning_rate)  
        
        return loss_history
        
    def predict(self, X):
        assert X.shape[-1] == self.W.shape[0], "X and W don't have compatible dimensions"
        return X @ self.W # np.dot(X, self.W)
    
 
if __name__ == "__main__":
    
    X, y = load_data(isStd=True)
    X_train, y_train, X_test, y_test = split_data(X, y, ratio=0.8)
    print(X_train.shape, y_train.shape)
    num_samples, num_features = X_train.shape
    
    clf = LinearRegression(num_features)
    loss = clf.fit(X=X_train, 
            y=y_train,
            num_iters=2000,
            learning_rate=0.01)
    
    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(loss)
    # plt.show()
    plt.savefig('cost_function.png')
    
    print(clf.W)
        