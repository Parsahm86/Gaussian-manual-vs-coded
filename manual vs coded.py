# ------------------------ Naive Bayes from scrach

import numpy as np
import time

class GaussionNaiveBayes:
    # fit
    def fit(self, x, y):
        n_sample, n_feature = x.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        self._mean = np.zeros((n_classes, n_feature), dtype=np.float32)
        self._var = np.zeros((n_classes, n_feature), dtype=np.float32)
        self._prior = np.zeros((n_classes), dtype=np.float32)

    # calculate mean, variance and prior
        for i,c in enumerate(self._classes):
            x_for_class_c = x[y == c]
            self._mean[i, :] = x_for_class_c.mean(axis=0)
            self._var[i, :] = x_for_class_c.var(axis=0)
            self._prior[i] = x_for_class_c.shape[0] / float(n_sample)
        
    # calculate likelihood
    def likelihood(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9  # to avoid division by zero
        exp = np.exp(-(x - mean) ** 2 / (2 * var)) # numerator
        denom = np.sqrt(2 * np.pi * var) # denominator
        return exp / denom


    # prediction method
    def predict(self, x):
        y_pred = [self._classify_sample(x) for x in x]
        return np.array(y_pred)
    
    # classification phase
    def _classify_sample(self, x):
        posteriors = []

        for i, c in enumerate(self._classes):
            pri = np.log(self._prior[i])
            post = np.sum(np.log(self.likelihood(i, x)))
            posterior = pri + post 
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    

# ------------ imports
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# load (synthesize) dataset
x, y = make_classification(n_samples=10000, n_features=20, n_classes=2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3 , random_state=42)
 
start_time = time.perf_counter()

gnb = GaussionNaiveBayes()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

end_time = time.perf_counter()
print(f"duration of the manual model was : {end_time - start_time}")
print(f"acc for manual model : {accuracy_score(y_test, y_pred) * 100}")
# ------------------
start_time = time.perf_counter()

sk_gnb = GaussianNB()
sk_gnb.fit(x_train, y_train)
y_pred_sk = sk_gnb.predict(x_test)

end_time = time.perf_counter()

print(f"duration of the sklearn model was : {end_time - start_time}")
print(f"acc for sklearn model : {accuracy_score(y_test, y_pred_sk) * 100}")