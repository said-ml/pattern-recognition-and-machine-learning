import numpy as np

class Distance:
    def __init__(self)->None:
        pass
    def calc_distance(self, arr1:np.ndarray, 
                      arr2:np.ndarray)->None:

        if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
            raise TypeError(
                f'the type{arr1.shape} or type{arr2.shape} not supported'
            )

        if not arr1.shape == arr2.shape:
            raise ValueError(
                f'the shape{arr1.shape} and shape{arr2.shape} not equal'
            )
        raise NotImplementedError(
            'the distance calculation method is not implemented, must be implemented in subclasses'
        )

    def __str__(self):
        return self.__class__.__name__

    # method that return the lowest distance name
    @staticmethod
    def get_lowest_name_distance(distances):

        if not distances:
            return None
        return min(distances, key=lambda distance: str(distance))
class EuclideanDistance:
    def __init__(self)->None:
        # define initaliser of the parent class
        super(Distance).__init__()
    def calc_distance(self, arr1:np.ndarray, arr2:np.ndarray)->float:
        if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
            
            raise TypeError(
                f'the type{arr1.shape} or type{arr2.shape} not supported'
            )

        if not  arr1.ndim == arr2.ndim :
            raise ValueError( 
                f'the dimension {arr1.shape[1]} and dimension {arr2.shape[1]} are not compatible'
            )

        #dis_euc=np.square((arr1[1]-arr2[1]).T@(arr1[1]-arr2[1]))
        dis_euc=np.sqrt(((arr1 - arr2[:, np.newaxis]) ** 2).sum(axis=2))

        assert (dis_euc).all()>=0
        return dis_euc

class ManhattanDistance:
    def __init__(self)-> None:
        super(Distance).__init__()

    def calc_distance(self, arr1: np.ndarray, arr2: np.ndarray) -> float:

        if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
            raise TypeError(
                f'the type{arr1.shape} or type{arr2.shape} not supported'
            )

        if not arr1.ndim == arr2.ndim:
            raise ValueError(
                f'the dimension of {arr1.ndim} and the dimension {arr2.ndim} not equal'
            )

        return  np.abs(arr1 - arr2[:, np.newaxis]).sum(axis=2)
class CosineDistance:
    def __init__(self)-> None:
        super(Distance).__init__()

    def calc_distance(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
            if not isinstance(arr1, np.ndarray) or not isinstance(arr2, np.ndarray):
                raise TypeError(
                    f'the type{arr1.shape} or type{arr2.shape} not supported'
                )

            if not arr1.shape == arr2.shape:
                raise ValueError(
                    f'the shape{arr1.shape} and shape{arr2.shape} not equal'
                )

            norm_arr1, norm_arr2=np.linalg.norm(arr1, axis=0), np.linalg.norm(arr2, axis=0)

            if norm_arr1==0 or norm_arr2==0:
                 raise ValueError(
                     f'the {arr1} and {arr2} must not be a vector zero'
                 )

            prod_arr1_arr2=np.dot(arr1, arr2)

            # Calculate cosine similarity
            cosine_similarity=prod_arr1_arr2/(norm_arr1*norm_arr2)

            assert  not cosine_similarity.all() >1  # that assertion is sure due prod_uv < = norm_u * nor_v

            # define the distance
            distance=1-cosine_similarity

            return distance

if __name__ == '__main__':
    arr1=np.array([1, 2, 3, 4, 5])
    arr2=np.array([1, 2, -3, 4, -5])

    # create an instances of all classes
    euclidean= EuclideanDistance()
    manhattan= ManhattanDistance()
    cosine = CosineDistance()

    print("Euclidean Distance:", euclidean.calc_distance(arr1, arr2))
    print("Manhattan Distance:", manhattan.calc_distance(arr1, arr2))
    print("Cosine Similarity:", cosine.calc_distance(arr1, arr2))

    # manipulate some trivial assertions
    assert euclidean.calc_distance(arr1, arr1)==0
    assert manhattan.calc_distance(arr1, arr1)==0
    assert cosine.calc_distance(arr1, arr1)==0
    assert cosine.calc_distance(arr1, arr2).all() < 1

    distances = [euclidean, manhattan, cosine]
    lowest_name_distance = Distance.get_lowest_name_distance(distances)
    print(lowest_name_distance)  # Output: "CosineSimilarity"
