from PIL import Image
from random import random
import numpy as np
from cmath import inf
import math
import ast
import sys
import csv
import pickle


def p_net(A,x,wList,bList):
    A = np.vectorize(A)
    r = np.vectorize(round)
    a = []
    a.append(x)
    for i in range(1,len(wList)):
        a.append(A(a[i-1]@wList[i]+bList[i]))
    return a[len(a)-1]

def sigmoid(x):
    return 1/(1+math.e**(-x))

def gradSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def error(y,a):
    return .5*np.linalg.norm(y-a)**2

def makeSet(x):
    w = [None]
    b = [None]
    for i in range(len(x)-1):
        tempW = []
        tempB = []
        for n in range(x[i]):
            tempRowW = []
            for m in range(x[i+1]):
                tempRowW.append(random()*2-1)
            tempW.append(tempRowW)
        for m in range(x[i+1]):
            tempB.append(random()*2-1)
        w.append(np.array(tempW))
        b.append(np.array([tempB]))
    return (w,b)

def testNetwork(testSet,w,b):
    count = 0
    for x,y in testSet:
        test = p_net(sigmoid,x,w,b)
        if test[0].tolist().index(max(test[0])) != y[0].tolist().index(1):
            count+=1
    return count/len(testSet)*100

def backProp(w,b,train,layers,A,Agrad,rate):
    for i in range(300):
        for x,y in train:
            n = layers-1
            a = []
            dot = []
            delta = []
            a = [None]*layers
            dot = [None]*layers
            delta = [None]*layers
            a[0] = x
            for l in range(1,layers):
                dot[l] = a[l-1]@w[l]+b[l]
                a[l] = A(dot[l])
            delta[n] = Agrad(dot[n])*(y-a[n])
            for l in range(n-1,0,-1):
                delta[l] = Agrad(dot[l])*(delta[l+1]@w[l+1].transpose())
            for l in range(1,layers):
                b[l] = b[l] + rate*delta[l]
                w[l] = w[l] + rate*a[l-1].transpose()@delta[l]
        print("Epoch: " + str(i))
        print("Train error: " + str(testNetwork(train,w,b)))
        print("Test error: " + str(testNetwork(testSet,w,b)))
        print()
    return(w,b)

f1 = open('mnist_train.csv')
csvreader1 = csv.reader(f1)
f2 = open('mnist_test.csv')
csvreader2 = csv.reader(f2)

trainingSet = []
testSet = []
# count = 0
# for row in csvreader1:
#     if count==1:
#         break
#     count+=1
#     row = row[1:len(row)]
#     for i in range(28):
#         curr = ""
#         for j in range(28):
#             if row[i*28+j] != "0":
#                 curr += " "
#             else:
#                 curr += "*"
#         print(curr)
#     print()
    # output = [0]*10
    # output[int(row[0])] = 1
    # i = [int(x)/255 for x in row[1:len(row)]]
    # trainingSet.append((np.array([i]),np.array([output])))

# for row in csvreader2:
#     output = [0]*10
#     output[int(row[0])] = 1
#     i = [int(x)/255 for x in row[1:len(row)]]
#     testSet.append((np.array([i]),np.array([output])))

# output1 = open("trainingset.pkl","wb")
# pickle.dump(trainingSet,output1)
# output2 = open("testset.pkl","wb")
# pickle.dump(testSet,output2)

# output3 = open("currWB.pkl","wb")
infileTest = open("testset.pkl","rb")
infileTrain = open("trainingset.pkl","rb")

trainingSet = pickle.load(infileTrain)
testSet = pickle.load(infileTest)


(w,b) = makeSet([784,300,100,10])

(w,b) = backProp(w,b,trainingSet,4,sigmoid,gradSigmoid,.5)   

    
