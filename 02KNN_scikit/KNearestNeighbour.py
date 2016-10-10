import numpy as np
from heapq import heappush, heappop, heapify

class MinPriorityQueue:
    def __init__(self, lst=[]):
        self.minheap = lst
        self.heapify()

    def push(self, idx, val):
        pass

    def pop(self):
        return idx, val

    def heapify(self)

class KNearestNeighbourClassifier:
    def __init__(self, k, metrics="cosine_sim", algorithm="PriorityQueue"):
            self.metrics = metrics
            self.k = k
        self.training_samples = []
        self.training_class_labels = []

    def fit(self, training_feature_vectors, training_class_labels):
        self.training_samples = training_feature_vectors
        self.training_class_labels = training_class_labels

    def predict(self, fv):
        N = len(self.training_samples)
        # Calculate Similarity, O(N * D)
        similarity = np.zeros(N, dtype=np.float64)
        idx = 0
        for row in self.training_feature_vectors:
            similarity[idx] = self.cosine_similarity(row, fv)
            idx += 1
        # Find k highest similarity fv
        minPQ =

        return predicted_class_labels
    def cosine_similarity(self, vector1, vector2):
        similarity = 0
        return similarity

    def euculidean_distance(self, vector1, vector2):
        return distance

if __name__ == '__main__':
    a = np.array([], dtype=np.float64).reshape(0, 100)
    for idx  in xrange(10):
        np.append([idx] * 100, )
