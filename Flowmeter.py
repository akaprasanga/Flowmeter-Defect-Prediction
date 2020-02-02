import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_dataset(path):
    dataset = np.loadtxt(path)
    return dataset

dataset = load_dataset("./datasets/Meter A")
dataset = shuffle(dataset)

training_features = dataset[:, :-1]
training_labels = dataset[:, -1]

training_features = (training_features - np.min(training_features)) / (np.max(training_features) - np.min(training_features))
X_train, X_test, y_train, y_test = train_test_split(training_features, training_labels, test_size=0.3, shuffle=True)

SVM_classifier = svm.SVC(kernel='poly')
SVM_classifier.fit(X_train, y_train)
SVM_predicted = SVM_classifier.predict(X_test)
SVM_accuracy = accuracy_score(y_test, SVM_predicted)

Decisiontree_classifier = DecisionTreeClassifier()
Decisiontree_classifier.fit(X_train, y_train)
Decisiontree_predicted = Decisiontree_classifier.predict(X_test)
Decisiontree_accuracy = accuracy_score(y_test, Decisiontree_predicted)


Randomforest_classifier = RandomForestClassifier()
Randomforest_classifier.fit(X_train, y_train)
Randomforest_predicted = Randomforest_classifier.predict(X_test)
Randomforest_accuracy = accuracy_score(y_test, Randomforest_predicted)

MLP_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, verbose=False, learning_rate_init=0.001)
MLP_classifier.fit(X_train, y_train)
MLP_predicted = MLP_classifier.predict(X_test)
MLP_accuracy = accuracy_score(y_test, MLP_predicted)

print("SVM Calssifier accuracy : ", SVM_accuracy*100)
print("Decision Tree Calssifier accuracy : ", Decisiontree_accuracy*100)
print("Randomforest Calssifier accuracy : ", Randomforest_accuracy*100)
print("MLP Calssifier accuracy : ", MLP_accuracy*100)
