import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv('house_price.csv')


# print(df['SalePrice'].describe())
# sns.histplot(df['SalePrice'])

def load_advanced_house_data(isStd=False):
    data = pd.read_csv('house_price.csv')
    # X = data['GrLivArea']
    
    X = data[['OverallQual', 'GrLivArea']]
    y = data['SalePrice']
    
    if isStd:
        X = (X - X.mean()) / X.std()
        # X = np.column_stack((np.ones(X.shape[0]), X))
        y = (y - y.mean()) / y.std()
        
        # print(len(X))
    return X, y

def load_simple_house_data(filename, isStd=False):
    data = pd.read_csv(filename, sep=",", index_col=False)
    data.columns = ["housesize", "rooms", "price"]
    X = data[["housesize", "rooms"]]
    y = data["price"]
    if isStd:
        X = (X - X.mean()) / X.std()
        y = (y - y.mean()) / y.std()
        X = np.column_stack((np.ones(X.shape[0]), X))
        
    return X, y
        

def split_data(X, y, ratio=0.8):
    # split = int(len(X) * ratio)
    split = len(X) - 625
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    
if __name__ == "__main__":
    
    X, y = load_simple_house_data('house.txt', True)
    X_train, y_train, X_test, y_test = split_data(X=X, y=y)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    