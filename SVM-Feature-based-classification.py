# Read an excel file of features and labels and train an svm classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import make_scorer
from sklearn.svm import SVC
import joblib

class Feature_based_classification:
    def __init__(self, filename):
        self.filename = filename
        self.data = pd.read_excel(self.filename)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=0)
        self.clf = SVC(kernel='rbf', C=1, gamma=0.1)
        self.balance_train_data()

    
    def balance_data(self, X, y):
        # Get class label count
        class_count = y.value_counts()
        # Get the index of the class with the least samples
        min_index = class_count.index[0]
        # Get the number of samples for the class with the least samples
        min_count = class_count.values[0]
        # Get the index of the class with the most samples
        max_index = class_count.index[-1]
        # Get the number of samples for the class with the most samples
        max_count = class_count.values[-1]
        # Get the number of samples to add
        add_count = max_count - min_count
        # Get the number of samples to remove
        remove_count = min_count
        # Get the indexes of the samples to remove
        remove_index = y[y == min_index].index
        # Get the indexes of the samples to add
        add_index = y[y == max_index].index
        # Get the indexes of the samples to remove
        remove_index = y[y == min_index].index
        # Get the indexes of the samples to add
        add_index = y[y == max_index].index
        # Get the indexes of the samples to remove
        remove_index = y[y == min_index].index
        # Get the indexes of the samples to add
        add_index = y[y == max_index].index
        # Get the indexes of the samples to remove
        remove_index = y[y == min_index].index
        # Get the indexes of the samples to add
        add_index = y[y == max_index].index
        # Get the indexes of the samples to remove
        remove_index = y[y == min_index].index
        # Get the indexes of the samples to add
        add_index = y[y == max_index].index
        # Get the indexes of the samples to remove
        remove_index = y[y == min_index].index
        # Get the indexes of the samples to add
        add_index = y[y == max_index].index

    def plot_roc_curve(self, y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

    def plot_precision_recall_curve(self, y_test, y_pred):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        plt.plot(precision, recall)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve')
        plt.show()
    
    def plot_precision_recall_fscore_support(self, y_test, y_pred):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
        plt.plot(precision, recall)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve')
        plt.show()
    
    def balance_train_data(self):
        self.X_train_balanced, self.y_train_balanced = self.balance_data(self.X_train, self.y_train)
     
    def train(self):
        self.clf.fit(self.X_train_balanced, self.y_train_balanced)
    
    def test(self):
        y_pred = self.clf.predict(self.X_test)
        print('Accuracy:', accuracy_score(self.y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(self.y_test, y_pred))
        print('Classification Report:')
        print(classification_report(self.y_test, y_pred))
        print('ROC Curve:')
        self.plot_roc_curve(self.y_test, y_pred)
        print('Precision-Recall Curve:')
        self.plot_precision_recall_curve(self.y_test, y_pred)
        print('F1 Score:', f1_score(self.y_test, y_pred))
        print('Precision Score:', precision_score(self.y_test, y_pred))
        print('Recall Score:', recall_score(self.y_test, y_pred))
        print('Precision-Recall F1-Score:', precision_recall_fscore_support(self.y_test, y_pred))
        print('Average Precision Score:', average_precision_score(self.y_test, y_pred))
    
    def save_model(self, filename):
        joblib.dump(self.clf, filename)
    
    def load_model(self, filename):
        self.clf = joblib.load(filename)
    
    def predict(self, X):
        return self.clf.predict(X)
    

if __name__ == "__main__":
    filename = '../data/feature_based_classification.xlsx'
    model = Feature_based_classification(filename)
    model.train()
    model.test()
    model.save_model('../model/feature_based_classification.pkl')
    model.load_model('../model/feature_based_classification.pkl')
    X = model.X_test.iloc[0:1, :]
    print(model.predict(X))

    



