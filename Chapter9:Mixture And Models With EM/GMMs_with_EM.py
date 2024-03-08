import numpy as np
from typing import List

# define  a helper function
def is_symmetric_positive_definite(matrix: np.ndarray) -> bool:
    #assert matrix.shape[0] == matrix.shape[1] # assert  is that matrix is a squared matrix

    IS_symmetric_positive_definite=True

    # make sure is symmetric
    if not np.allclose(matrix, matrix):
        IS_symmetric_positive_definite=False

    # make sure that eignvalues are positive ==> the matrix is positive
    eignvalues = np.linalg.eigvals(matrix)    # extracting the eignvalues from the matrix

    if not eignvalues.all()>=0:
        IS_symmetric_positive_definite=False

    return IS_symmetric_positive_definite


class GaussianMixtureModels:
    def __init__(self,
                 n_iterations:int=5,
                 n_clusters:int=None,
                 mixing_coeffs: List[float]=None,
                 seed:int=42,
                 threshold:float=0.001,
                 algorithm:str='Expectation-Maximization'
                 )->None:

                  '''

                  :param n_iterations: the maximum number of iterations
                  :param n_clusters: the number of the clusters that we assume the data is generate from it, it the number of components
                  :param mixing_coeffs: a coefficents between 0 and 1 for each cluster represents the probabilities of each cluster soft assignment
                  :param seed: make the output similar
                  :param threshold: that argument represent error tolerance that we can admit
                  :param algorithm: method to training the parameters of gaussian mixture models
                  '''

                  supported_algorithms = ['Expectation-Maximization',
                                          'Kmeans Initialization for Expectation-Maximization'
                                          'Variational Inference',
                                          'Bayesian Inference',
                                          'Non Parametric Methods',
                                          'Gradient Descent']

                  if algorithm not in supported_algorithms:
                      raise ValueError(
                          f'the :{algorithm} is not supported by this model, try one '
                          f'of this list: {supported_algorithms}')

                  self.n_iterations=n_iterations
                  np.random.seed(seed)

                  if n_clusters is None:
                      raise ValueError(
                          f'n_components must not be None')

                  self.n_clusters = n_clusters

                  if mixing_coeffs is None:
                      mixing_coeffs=np.ones((self.n_clusters,1))/self.n_clusters

                  self.mixing_coeffs = mixing_coeffs

                  assert self.mixing_coeffs.all()>=0
                  if  not self.mixing_coeffs.sum()==1:
                      raise ValueError(
                          f'the sum of probabilities {self.mixing_coeffs.sum()} not equal to 1')

                  if threshold is None:
                      raise ValueError(
                          f'tolerance must not be {threshold} ')

                  assert threshold>0
                  self.threshold = threshold
    def fit(self,X:np.ndarray, y=None,
                      random_initialization:bool=True)->None:

                      '''
                      :param self: fitted gaussian mixture model
                      :param X: training data
                      :param y: we set it t None (those are unsupervised algorithm)
                      :return: None
                      '''

                      assert y is None # the fit(.) in most of time has two arguments samples or data X
                      # and targets(or labels) y, y is optional and not required in the case of unsupervised learning algorithms

                      # extractiong the samples and features
                      n_samples, n_features = X.shape
                      # n_samples are rows of data and features are the columns

                      if random_initialization:

                          # in ML random initialization result to better performance and avoid some issues arise the symetric
                          self.means = np.random.rand(self.n_clusters, n_features)

                          def covariances(n_clusters, n_features)->np.ndarray:
                              covariances = np.empty((n_clusters, n_features, n_features))
                              for i in range(n_clusters):
                                  random_matrix = np.random.rand(n_features, n_features)
                                  covariance_matrix = np.dot(random_matrix,
                                                             random_matrix.T)
                                  covariances[i] = covariance_matrix
                              return covariances
                          self.covariances=covariances(self.n_clusters, n_features)

                      else:
                          self.means = np.zeros((self.n_clusters, n_features))
                          self.covariances = np.array([np.eye(n_features)] * self.n_clusters)

                      # the covariances matrix must be symmetric positive definite , for x  , x.T@covariancesx>=0
                      if not is_symmetric_positive_definite(self.covariances):
                              raise ValueError(
                                  f'{self.covariances} must be symmetric positive and definite ')

                      #if self.algorithm=='Expectation-Maximization':

                      '''
                      for good understanding of the EM algorithm for Gaussian Mixture Models
                      check out the `pattern recognition and machine learning` bishop's book, page 439
                      
                      ------------- pseudo EM algorithm for Gaussian Mixture --------------------------------------------------
                      |                                                                                                       |
                      |  1==> parameters initialization: means mu_{k}, covariances sigma_{k}, mixing-coefficients p_{k}       |
                      |                                                                                                       |          
                      |  2==> E-Step(stand for Expectation step):                                                             |
                      |       Evaluation of the responsibilities for the current parameters:                                  |
                      |              gamma(Z_{nk})=(p_{k}N(X_{n}/mu_{k}, sigma{k)\sum_{j=1}^{K}N(X_{n}/mu_{j}, sigma{j})      |
                      |                                                                                                       |
                      | 3==> M-Step(stand for Maximum Likelihood step):                                                       |
                      |      Re-estimating the parameters using the current responsibilities                                  |
                      |      mu_{k}^{new}=1/(N_{k}sum_{n=1}^{k}N(X_{n}/mu_{k}), sigma_{k})                                    |
                      |                                                                                                       |
                      | 4==> log-likelihood evaluation:                                                                       |
                      |      ln(p(X/mu, sigma)=sum_{n=1}^{N}{sum_{k=1}^{K}p_{k}N(X_{n}/mu_{k}, sigma{k}}                      |  
                      |                                                                                                       |
                      | check out the convergence: here convergence criteria is a threshold that not are note  any more       |
                      |  the current log-likelihood and the previous log-likelihood or the number of iterations         |     | 
                      |  maximale, in any step of the EM-algorithm is satisfied we brak the algorith and predict the           |
                      |  otherwise we perform step 2 and so on, ... till to the convergence                                         | 
                      _________________________________________________________________________________________________________   
                      
                      for more information check out :https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738                                                                      
                      '''

                      for _ in range(self.n_iterations):

                                  prev_log_likelihood =None  # for starting
                                  # E-step: Expectation
                                  responsibilities =self.update_responsibilities(X)

                                  # M-step: Maximization
                                  self.update_parameters(X, responsibilities)

                                  # Compute the log-likelihood
                                  log_likelihood=self.compute_log_likelihood(X)

                                  # Check for convergence
                                  if prev_log_likelihood is not None and (np.abs(prev_log_likelihood-log_likelihood) < self.threshold):
                                  # you can change the convergence creteria by checking the differences between responsibilities
                                      break

                                  prev_log_likelihood=log_likelihood

    staticmethod
    def update_responsibilities(self, X:np.ndarray)->np.ndarray:
                                  n_samples = X.shape[0]
                                  responsibilities = np.zeros((n_samples, self.n_clusters))

                                  for i in range(self.n_clusters):
                                      diff = X - self.means[i]
                                      exponent = -0.5 * np.sum(np.dot(diff, np.linalg.inv(self.covariances[i])) * diff, axis=1)
                                      coef = 1. / np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.covariances[i]))
                                      responsibilities[:, i] = coef * np.exp(exponent)
                                  print(responsibilities.shape, self.mixing_coeffs.shape)

                                  responsibilities = responsibilities@self.mixing_coeffs
                                  responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

                                  return responsibilities


    def update_parameters( self, X:np.ndarray, responsibilities:np.ndarray)->np.ndarray:
                                  n_samples = X.shape[0]

                                  # Update weights
                                  self.weights = np.mean(responsibilities, axis=0)

                                  # Update means
                                  weighted_sum = np.dot(responsibilities.T, X)
                                  self.means = weighted_sum / np.sum(responsibilities, axis=0)[:, np.newaxis]

                                  # Update covariances
                                  for i in range(self.n_clusters):
                                      diff = X - self.means[i]
                                      weighted_diff = responsibilities[:, i][:, np.newaxis] * diff
                                      self.covariances[i] = np.dot(weighted_diff.T, diff) / np.sum(responsibilities[:, i])


    def compute_log_likelihood(self, X:np.ndarray)->np.ndarray:

        log_likelihood = 0
        for k in range(self.n_clusters):
            diff = X - self.means[k]
            exponent = -0.5 * np.sum(np.dot(diff, np.linalg.inv(self.covariances[k])) * diff, axis=1)
            log_likelihood += np.sum(
                np.log(self.weights[k] * np.exp(exponent) / np.sqrt(np.linalg.det(self.covariances[k]) + 1e-12))) # 1e-12 add it fo compuataion stabilities
        return log_likelihood

    def plot_clustering(self, X:np.ndarray)->None:
        try:
            import matplotlib.pyplot as plt
            pass
        except ImportError:
            raise ModuleNotFoundError(
                'install matplotlib throgh command-line : pip install matplotlib'
            )
    def predict(self, X_test:np.ndarray)->np.ndarray:

                            responsibilities = self.update_responsibilities(X_test)
                            return np.argmax(responsibilities, axis=1)









if __name__ == '__main__':
    np.random.seed(0)

    # Define a GMM object
    GMM=GaussianMixtureModels(
        n_clusters=3,
        n_iterations=6,
        algorithm='Expectation-Maximization',
        threshold=0.005
    )

    # create som fake data
    np.random.seed(42)
    n_samples = 100    n_features = 2
    centers = np.array([[1, 1], [-1, -1]])
    X = np.concatenate([np.random.randn(n_samples, n_features) + center for center in centers])

    # train it
    GMM.fit(X)
    GMM.predict(X)
