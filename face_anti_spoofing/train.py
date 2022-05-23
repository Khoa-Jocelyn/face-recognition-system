import os
import cv2
import  time
import random
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
import pickle
import json

data_path = "raw_data/"

label2id = {}
def load_data(data_path):
    global label2id
    X = []
    y = []
    for id, file in enumerate(os.listdir(data_path)):
        label = file.split(".")[0]
        data = pd.read_csv(os.path.join(data_path, file))
        dataset = data.iloc[:, 1:].values
        step = 1
        for i in range(step, len(dataset)):
            X.append(dataset[i - step:i,:])
            y.append(label)
            i = i + step
        label2id[label] = len(dataset)
    print(json.dumps(label2id, indent=4))
    return X, y


print("Data Loading...")
X, y = load_data(data_path)
print("Data Processing...")
X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[2])
print("==>X: %s \n==>y: %s" % (X.shape, y.shape))

from sklearn import svm
print("Training Started")
model = svm.SVC(kernel='linear', C=1, probability=True)
model.fit(X, y)

# Saving model
with open("class/classifier.pkl", 'wb') as outfile:
    pickle.dump((model, list(label2id.keys())), outfile)
    print("Traiing Done!")

from sklearn.metrics import confusion_matrix
y_pred = model.predict(X)
conf_matrix = confusion_matrix(y, y_pred)
print(conf_matrix)
