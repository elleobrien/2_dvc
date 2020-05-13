from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
import json
import os
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
depth = 10
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

acc = clf.score(X_test, y_test)
with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc}, outfile)


