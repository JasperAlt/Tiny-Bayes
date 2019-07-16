from math import *                     # Naive Bayes Classifier
import numpy as np, pandas as pd, sys  # This runs a naive bayes classifier on the spambase dataset.
                                       # Training and test sets are created from a random shuffle of the data
                                       # Results are averaged over a number of runs given at the command line
###################
#     Setup
###################              # Take number of runs from command line
                                 # Spambase dataset from file
runs             =  int(sys.argv[1]) if len(sys.argv) > 1 else 1
emails           =  pd.read_csv('spambase/spambase.data', header = None).sample(frac=1)
(half, C, zero)  =  (len(emails) // 2,  [[0,0],[0,0]],  1/pow(10,60))
                                 # ^ Half data length, empty confusion matrix, near zero value
def P(i, train): return i + pow(-1,i) * len(train[train[57] == 0])/(0.0 + len(train))
                                 # Helpers for priors and normal
def N(x, mu, sigma): return (1.0/(sqrt(2.0*pi)*sigma)) * pow(e, (-(x-mu)*(x-mu))/(2.0*sigma*sigma))
                                 # Compute mean or std deviation on features of each class
def stat(name, train): return [[i for i in (train[train[57]==c].mean(axis=0) if name is "mean" else train[train[57]==c].std(axis=0)).replace(0,zero)] for c in [0,1]]
                                 # Classification function sums logs of priors, feature probabilities each class, takes argmax
def classify(X, means, devs): return np.argmax([[log(priors[c_j]) + sum([log(N(X[i], means[c_j][i], devs[c_j][i]) or zero) for i in range(len(X)-1)])] for c_j in [0,1]])
                                 # "Take the argmax of the log of the class prior plus the sum of logs of feature probabilities for each class"
###################
#   Experiment
###################
                                 # Each run:
for run in range(runs):
                                 # 1. split data and shuffle for next run
  (train, test, emails) = (emails[:half],  np.array(emails[half:]),  emails.sample(frac=1))
                                 # 2. compute priors, means and deviations
  (priors, means, devs) = ([P(0, train),P(1, train)],  stat("mean", train),  stat("std", train))
                                 # 3. classify test data
  for i in range(len(test)): C[int(classify(test[i],means,devs))][int(test[i][57])] += 1.0
                                 # "Tally in the confusion matrix row of the classification, column of the actual class"
###################
#    Reporting
###################
                                 # Average, round, and show confusion matrix & derived stats
(TN,FN,FP,TP) = (round(item/runs,2) for item in C[0] + C[1])
print("\n\tConfusion Matrix\n\nClassed\t  0  "+str(TN)+"\t"+str(FN)+"\n\t  1  "+str(FP)+"\t"+str(TP)+"\n\n\t\t0\t1\n\t\t  Actual\n")
print("\nAccuracy\t"+str((TN+TP)/(FN+FP+TP+TN))+"\nPrecision\t"+str(TP/(FP+TP))+"\nRecall\t\t"+str(TP/(TP+FN))+"\n")

              # MINIMIZED TO CONSERVE PRECIOUS NATURAL RESOURCES
