import numpy as np
from typing import Optional

from clustering.distance import EuclideanDistance
from clustering.distance import ManhattanDistance
from clustering.distance import CosineDistance


import numba as nb


class KMeansBase(object):
    def __init__(self,
                 n_clusters:int=8,
                 max_iter:int=300,
                 tol:float=1e-4,
                 intialization:str='random',
                 distance:Optional[str]='euclidean',
                 seed:int=42)->None:

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        np.random.seed(seed) # to make output stable and unique

        if not distance in ['euclidean', 'manhattan', 'cosine_similarity']:
            raise ValueError(f'the distance {distance} is not supported')

        self.distance=distance

    def elbow(self,X:np.ndarray)->None:
        pass

    def initialize_centroids(self, X:np.ndarray)->np.ndarray:

        if len(X)<=self.n_clusters:
            raise ValueError(f'the number {self.n_clusters} of clusters must be less than the size {len(X)}of samples')

        indices = np.random.choice(len(X), self.n_clusters, replace=False)

        return X[indices]

    def assign_to_clusters(self, X:np.ndarray, centroids:np.ndarray)->float:

        #if self.distance == 'euclidean' or self.distance==None:
        #distances = EuclideanDistance().calc_distance(X, centroids)#[:, np.newaxis])
        #distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))

        #elif self.distance == 'manhaten':
        distances = ManhattanDistance().calc_distance(X ,centroids)

        #else:
            #distances = CosineDistance().calc_distance((X - centroids[:, np.newaxis]))

        return np.argmin(distances, axis=0)

    def update_centroids(self, X:np.ndarray, labels:int)->np.ndarray:
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        print(centroids)
        for i in range(self.n_clusters):
            centroids[i] = np.mean(X[labels == i], axis=0)
        return centroids


    #@nb.jit(nopython=True)
    def fit(self, X:np.ndarray, y=None)->None:
    # we set the target to None due this is KMeans is unsupervised algorithm

        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self.assign_to_clusters(X, self.centroids)
            new_centroids = self.update_centroids(X, labels)
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
        self.labels = labels

    def plot_clusters(self, X:np.ndarray, fig_size:tuple[int]=(8, 8))->None:
      try:

        import matplotlib.pyplot as plt
        plt.figure(figsize=fig_size)
        for i in range(self.n_clusters):
            plt.scatter(X[self.labels == i][:, 0], X[self.labels == i][:, 1], label=f'Cluster {i+1}')
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', color='black', label='Centroids')
            plt.title('K-means Clustering')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()

      except ModuleNotFoundError:
          print('Please install matplotlib, type in command-line: pip install matplotlib')





class KMeans(KMeansBase):
    pass

class MiniBatchKMeans(KMeansBase):
    def __init__(self, batch_size:int=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def fit(self, X:np.ndarray, y=None)->None:
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            batch_indices = np.random.choice(len(X), self.batch_size, replace=False)
            X_batch = X[batch_indices]
            labels = self.assign_to_clusters(X_batch, self.centroids)
            new_centroids = self.update_centroids(X_batch, labels)
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
        self.labels = self.assign_to_clusters(X, self.centroids)

class OnlineKMeans(KMeansBase):
    def fit(self, X:np.ndarray, y=None)->None:
        self.centroids = self.initialize_centroids(X)
        for x in X:
            labels = self.assign_to_clusters(x[np.newaxis, :], self.centroids)
            new_centroids = self.update_centroids(x[np.newaxis, :], labels)
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
        self.labels = self.assign_to_clusters(X, self.centroids)

# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = np.random.rand(100, 2)

    # Instantiate and fit KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    print("KMeans Labels:", kmeans.labels)

    # Plot KMeans clusters
    kmeans.plot_clusters(X)

    # Instantiate and fit MiniBatchKMeans
    minibatch_kmeans = MiniBatchKMeans(n_clusters=3)
    minibatch_kmeans.fit(X)
    print("MiniBatchKMeans Labels:", minibatch_kmeans.labels)

    # Plot MiniBatchKMeans clusters
    minibatch_kmeans.plot_clusters(X)

    # Instantiate and fit OnlineKMeans
    online_kmeans = OnlineKMeans(n_clusters=3)
    online_kmeans.fit(X)
    print("OnlineKMeans Labels:", online_kmeans.labels)

    # Plot OnlineKMeans clusters
    online_kmeans.plot_clusters(X)
