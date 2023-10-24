import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
# loading breast cancer dataset
from sklearn.datasets import load_breast_cancer


class LinearLeastSquare:
    
    def __init__(self, W=None):
        self.W=W

    #@staticmethod
    def pinv(self, A):
        return inv(X.T@X)@X.T

    def fit(self, X, y):
        # we add the basis
        X_bias= np.hstack((np.ones((X.shape[0], 1)), X))
        #self.W=self.pinv(X)@y
        self.W=np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        print(self.W)
        
    def class_proba(self, X):
        self.W=self.W.reshape(3,1)
        # we add the basis
        X=np.hstack((np.ones((X.shape[0], 1)), X))
        return X@self.W
        print('ghh',X.shape, self.W.shape)
        #print(X@self.W)
    def predict(self, X):
        self.W=self.W.reshape(3,1)
        print(X.shape, self.W.shape)
           # we add the basis
        #X=np.hstack((np.ones((X.shape[0], 1)), X))
        class_proba=self.class_proba(X)
        # assert that is really probabilirty
        #assert class_proba<=1.
        y_pred=np.where(class_proba<.5, 0, 1)
        return y_pred

    def accuracy(self, X, y):
        y_pred=self.predict(X)
        return np.mean(y_pred==y)
        

    def boudaries_show(self, X, y):
        # Define the decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        Z = self.W[0] + self.W[1] * xx + self.W[2] * yy

        # Plot the decision boundary
        plt.contour(xx-6, yy, Z, levels=[0], colors='red')

        # Plot the data points
        plt.scatter(X[:, 0], X[:, 1], c=y)

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Linear Least Squares Classification of Breast Cancer Data')
        plt.show()


if __name__=='__main__':
        # load the cancer data
        data=load_breast_cancer()
        # select only the first two features
        X=data.data[:, :2]
        y=data.target

        # create the model
        model=LinearLeastSquare()
        # train the model
        model.fit(X, y)
        model.class_proba(X)
        # make predictions
        y_pred=model.predict(X)

        # evalute the model
        accuracy=model.accuracy(X, y)
        print('accuracy=', accuracy)

        # plot the decision boundary and the data points
        model.boudaries_show(X, y)
        plt.show()
