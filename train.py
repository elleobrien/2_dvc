from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
import json
import os
import numpy as np

import matplotlib.pyplot as plt 
import scikitplot as skplot

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")


# Fit a model
depth = 10
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train,y_train)

# Get overall accuracy
acc = clf.score(X_test, y_test)

# Get precision and recall
y_score = clf.predict(X_test)
prec = precision_score(y_test, y_score)
rec = recall_score(y_test,y_score)


with open("metrics.json", 'w') as outfile:
        json.dump({ "accuracy": acc, "precision":prec,"recall":rec}, outfile)

# Make a Precision-recall curve
probas = clf.predict_proba(X_test)
skplot.metrics.plot_precision_recall_curve(y_test, probas)
plt.savefig("pr_curve.png")

