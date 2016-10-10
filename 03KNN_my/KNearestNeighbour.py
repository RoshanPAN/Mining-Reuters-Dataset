import numpy as np
from heapq import heappush, heappop, heapify

class MinPriorityQueue:
    def __init__(self, lst=[]):
        self.minheap = lst
        heapify(self.minheap)

    def push(self, idx, val):
        heappush(self.minheap, (idx, val))

    def pop(self):
        return heappop(self.minheap)


class KNeighborsClassifier:
    def __init__(self, k, metrics="cosine_similarity", algorithm="brute_force"):
        self.metrics = metrics
        self.k = k
        self.training_sample_fv = []
        self.training_class_labels = []

    def fit(self, training_feature_vectors, training_class_labels):
        self.training_sample_fv = training_feature_vectors
        self.training_class_labels = training_class_labels

    def predict(self, fv):
        N = len(self.training_sample_fv)
        # Calculate Similarity, O(N * D)
        similarity = np.zeros(N, dtype=np.float64)
        idx = 0
        for row in self.training_sample_fv:
            similarity[idx] = self.cosine_similarity(row, fv)
            idx += 1
        # Find k highest similarity fv
        tmp = []
        for idx, sim in enumerate(similarity):
            tmp.append((sim, idx))
        tmp.sort(cmp=self._compare, reverse=True)
        topK = tmp[:self.k]
        # Get the topk
        topK_idxs = map(lambda x: x[1], topK)
        # Combine Topics, if there is no topics
        preds = set(self.training_class_labels[topK_idxs[0]])
        # print " First  idx: ", idx, set(self.training_class_labels[topK_idxs[0]]), "   Prediction:",preds
        for idx in topK_idxs[1:]:
            tmp = set(self.training_class_labels[idx])
            preds = preds.intersection(tmp)
            # print "   idx: ", idx, tmp, "   Prediction:",preds
        if len(preds) == 0:
            # print "[Predictor] No Intersection"
            preds = self.training_class_labels[topK_idxs[0]]
            return list(preds)
        else:
            # print "[Predictor] Intersection of K neighbors."
            return list(preds)


    def _compare(self, x, y):
        if x[0] > y[0]:
            return 1
        elif x[0] < y[0]:
            return -1
        else:
            if x[1] > y[1]:
                return -1
            elif x[1] < y[1]:
                return 1
            else:
                return 0

    def cosine_similarity(self, vector1, vector2):
        assert len(vector1) == len(vector2)
        length = len(vector1)
        xy = np.sum(np.multiply(vector1, vector2))
        x = np.sqrt(np.sum(np.square(vector1)))
        y = np.sqrt(np.sum(np.square(vector2)))
        similarity = xy / x  / y
        return similarity

    def __str__(self):
        s = ""
        s += "Training Dataset:  " + str(self.training_sample_fv) + "\n"
        s += "Class Labels:  " + str(self.training_class_labels) + "\n"
        return s


if __name__ == '__main__':
    # For test
    import random
    a = np.array([], dtype=np.float64).reshape(0, 100)
    for idx  in xrange(1, 11):
        a = np.concatenate((a, [[random.randint(1, 10)  for _ in xrange(100)]]), axis=0)
    clf = KNeighborsClassifier(k=5)
    b = np.arange(100).reshape(10, 10)
    print b
    clf.fit(a, b)
    fv = np.array([5] * 100, dtype=np.float64)
    print fv
    print clf.predict(fv)


