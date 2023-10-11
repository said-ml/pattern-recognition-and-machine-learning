import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as gaussian
import matplotlib.pyplot as plt
from  scipy.stats import norm as normal_distribution
import scipy
#if get_ipython:
try:
    import numba as nb
except ModuleNotFoundError:
    print('tape  pip insntall numba  in your terminal or cmd to install numba')
    import numba as nb

import sys
if not sys.warnoptions:
   import warnings
   warnings.filterwarnings('ignore')

f=lambda x,a:a[0]+a[1]*x+0.1*np.random.random()

class bayesianlinearregression:

      def __init__(self, alpha=2. , beta=1, EPOCHS=1001):
          self.alpha=alpha
          self.beta=beta
          self.EPOCHS=EPOCHS

     #####----Compute The Posterior Mean And Covariance----#####

      @nb.jit(nopython=False)
      # decorator to compile the code(jit stands for just in time)
      def fit(self, X_train, y_train, mean_0=None, S_0=None):
          # define the basis-function Phi
          Phi=np.column_stack([ np.ones_like(X_train), X_train])
          N = X_train.shape[0]
          if mean_0 is None:
             mean_0=np.zeros(N)
          if S_0 is None:
              S_0=self.alpha*np.eye(N)
          S_N =inv(inv(S_0)+self.beta*Phi.T@Phi)
          mean_N=S_N@(inv(S_0)@mean_0+self.beta*Phi.T@y_train)
          return mean_N, S_N


      def predict(self, x_test):
          # give the basis-function
          Phi_test=np.column_stack([np.ones_like(x_test), np.ones(x_test)])
          X_train=np.linspace(-1, 1, 29)
          y_train=f(X_train, np.array([-.3, .5]))
          mean_N, S_N=self.fit(X_train,  y_train)
          #w=multivariate_normal.pdf(mean=mean_N.ravel(), cov=S_N)
          y_pred=mean_N@Phi_test
          return y_pred

      @nb.jit(nopython=False)
      def ploting_results(self):
          mean_N, S_N=np.array([0.0, 0.0]), np.array([[1., .0], [0.0, 1.]])
          x=np.linspace(-1, 1, 29, dtype=np.int64)
          Phi = np.column_stack([np.ones_like(x),x])
          y=f(x,np.array([-.3, .5]))
          w_0, w_1 = np.mgrid[-1:1:.07, -1:1:.07]
          # w=(w_0, w_1)
          w = np.dstack((w_0, w_1))
          gauss = gaussian(mean=mean_N.ravel(), cov=S_N)
          fig = plt.figure()
          ax = fig.add_subplot(111)
          ax.contourf(w_0, w_1, gauss.pdf(w))
          for epoch in range(self.EPOCHS):
               if epoch==0:
                 gauss= gaussian(mean=mean_N.ravel(), cov=S_N)
                 fig = plt.figure()
                 ax =fig.add_subplot(111)
                 ax.contourf(w_0, w_1, gauss.pdf(w))
                 plt.scatter(-0.3, 0.5, marker="x")
                 plt.show()
                 # define the likelihood function
                 # first we're getting te basis function
                 #likelihood=normal_distribution.pdf(y, loc=w.T.reshape(29, 58)@Phi.reshape(58, 1), scale=np.sqrt(1/self.beta)) #reshape(2, 29)
                 #plt.contourf(w_0, w_1, likelihood)
                 #for a in np.linspace(-4, 4, 10):
                     #plt.plot(x, a*x+np.random.random(), c='blue', alpha=.4)
               if  epoch in [1, 5 , 1000] :
                 #mean_N, S_N = self.fit(X_train=x, y_train=y, mean_0=mean_N.ravel(), S_0=S_N)
                 gauss= gaussian(mean=mean_N.ravel(), cov=S_N)
                 fig = plt.figure()
                 ax = fig.add_subplot(111)
                 #w = np.random.multivariate_normal(mean=mean_N.ravel(), cov=S_N, size=10)
                 #x = np.linspace(-1, 1, 29)

                 #ax.plot(x, f(x, mean_N), c="tab:blue", alpha=0.4)
                 #plt.scatter(x, f(x, np.array([-.3, .5])), label='actual_data', color='red')
                 #plt.scatter(x, y_pred)
                 ax.contourf(w_0, w_1, gauss.pdf(w))
                 #likelihood = scipy.stats.norm.pdf(y, loc=w.T.reshape(29, 58) @ Phi.T.reshape(58, 1), scale=np.sqrt(1 / self.beta))
                 #plt.contourf(w_0, w_1, likelihood)
                 plt.scatter(-0.3, 0.5, marker="x")
                 plt.show()
               mean_N, S_N = self.fit(X_train=x, y_train=y, mean_0=mean_N.ravel(), S_0=S_N)
               #plt.show()
          print(mean_N, S_N)

      def display_figure(self):
          # creating the sample data
          fig, axes=plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
          # plot the data on each subplot
          for i, ax in enumerate(axes.flat):
              #ax.plot(x, y)
              ax.set_title(f'tape your title here')
      # Adjust the spacing between subplots
      plt.tight_layout()
      # Display the figure
      plt.show()

if __name__=='__main__':
  baeslinear=bayesianlinearregression()
  baeslinear.ploting_results()
