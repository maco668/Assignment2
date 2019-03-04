# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:01:33 2019

@author: DMa
"""
import mlrose
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_covtype
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


#hasVC=True
hasVC=False
numFolds=5 # split training data into numFolds for CV

#data = load_iris()
#X=data.data
#y=data.target

X= fetch_covtype().data
y = fetch_covtype().target

#algo='random_hill_climb'
#algo='simulated_annealing'
algo='genetic_alg'

print(X[0])
total=580e3

trainScores=[]
testScores=[]
numIts=[]
#domain=np.linspace(0, 0.8, 5)
trainSize=0.8
print (trainSize)
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                    test_size = (1-trainSize), random_state = 3)
# Normalize feature data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One hot encode target values
one_hot = OneHotEncoder()

y_train_hot = one_hot.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = one_hot.transform(y_test.reshape(-1, 1)).todense()


domain=np.linspace(5, 50, 5)
for numIt in domain:
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

#    clf=MLPClassifier(activation='tanh', alpha=1e-05, batch_size='auto',
#                  beta_1=0.9, beta_2=0.999, early_stopping=False,
#                  epsilon=1e-08, hidden_layer_sizes=(100,),
#                  learning_rate='constant', learning_rate_init=0.0001,
#                  max_iter=200, momentum=0.9, n_iter_no_change=10,
#                  nesterovs_momentum=True, power_t=0.5, random_state=3,
#                  shuffle=True, solver='lbfgs', tol=0.00001,
#                  validation_fraction=0.1, verbose=False, warm_start=False)
    numIt=int(numIt) 
    numIts.append(numIt)
    print("max_iters = ",numIt)
#    clf = mlrose.NeuralNetwork(hidden_nodes = [100,], activation = 'tanh', \
#                                     algorithm = 'random_hill_climb', max_iters = numIt, \
#                                     bias = True, is_classifier = True, learning_rate = 0.1, \
#                                     early_stopping = True, max_attempts = 100, \
#                                     random_state = 3, restarts=1)
    
#    clf = mlrose.NeuralNetwork(hidden_nodes = [100,], activation = 'tanh', \
#                                     algorithm = 'simulated_annealing', max_iters = numIt, \
#                                     bias = True, is_classifier = True, learning_rate = 0.1, \
#                                     early_stopping = True, max_attempts = 100, \
#                                     random_state = 3)
    
    clf = mlrose.NeuralNetwork(hidden_nodes = [100,], activation = 'tanh', \
                                     algorithm = 'genetic_alg', max_iters = numIt, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     pop_size=500, early_stopping = True, mutation_prob=.2,\
                                     max_attempts = 50, \
                                     random_state = 3)
        
    clf.fit(X_train_scaled, y_train_hot)
    
#    y_train_pred = clf.predict(X_train_scaled)
#    
#    y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)
#    trainScores.append(y_train_accuracy)
#
#    print ("training size ",trainSize*total, "training scores: ", y_train_accuracy)
#    
#    y_test_pred = clf.predict(X_test_scaled)
#    
#    y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
#    
#    testScores.append(y_test_accuracy)
#    print('test score: ', y_test_accuracy)
#    
#   
#    elif hasVC==True:    
    
    print ('using cross_val_score helper function with ', numFolds, ' folds for training data')
    
    
    CV_training_scores = cross_val_score(clf, X_train_scaled, y_train_hot, cv=numFolds, scoring='accuracy')
    trainScores.append(CV_training_scores)
    realSize.append(trainSize*total)
    print ("training size ",trainSize*total, " CV training scores:", CV_training_scores.mean)
    
    
    
    y_test_pred = clf.predict(X_test_scaled)
    
    CV_y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)
    testScores.append(CV_y_test_accuracy)
    
    print('test score after VC')
    print(CV_y_test_accuracy)

plt.plot(numIts, trainScores, label='training accuracy', linewidth=2)
plt.plot(numIts, testScores, label='test accuracy', linewidth=2)
plt.xlabel('Max iterations')
plt.ylabel('Accuracy Score')
plt.title('ANN(' + algo + ') Learning Curve for Tree Cover Type')
plt.legend()
plt.grid(True)
#plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig("ANN_learning_curve(" + algo +")")
#







