# ------------------------------------------------------------------------------
# Name:        classify Images based on their labels and use them to train an SVM classifier
# Purpose:	   Images are used directly to the svm classifier, and the labels are used to train the classifier
#
# Author:      Morteza Heidari
#
# Created:     11/20/2021
# ------------------------------------------------------------------------------
# read the images and labels for the SVM classification
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class SVM(object):
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

    def load_data(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.jpg'):
                    image = cv2.imread(os.path.join(root, file))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, (100, 100))
                    self.train_data.append(image)
                    self.train_labels.append(root)

    def split_data(self):
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.train_data, self.train_labels, test_size=0.2)

    def scale_data(self):
        self.train_data = self.scaler.fit_transform(self.train_data)
        self.test_data = self.scaler.transform(self.test_data)

    def train(self):
        self.model.fit(self.train_data, self.train_labels)

    def test(self):
        predictions = self.model.predict(self.test_data)
        print('Accuracy: {}'.format(accuracy_score(self.test_labels, predictions)))

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)

    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (100, 100))
        image = self.scaler.transform([image])
        return self.model.predict(image)


if __name__ == '__main__':
    svm = SVM()
    svm.load_data('dataset')
    svm.split_data()
    svm.scale_data()
    svm.train()
    svm.test()
    svm.save_model('svm.pkl')
    svm.load_model('svm.pkl')
    image = cv2.imread('dataset/0/0_1.jpg')
    print(svm.predict(image))
    print("All DONE!")
