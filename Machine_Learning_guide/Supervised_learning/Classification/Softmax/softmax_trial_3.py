#!/usr/bin/env python
# coding: utf-8
############################################################################################ SOFTMAX Trials 3
""" It must be RUN from terminal with following command (change parameters as you like): 
        $python softmax_trial_3.py -e 5 -lr 0.05 -bs 20 -r 0.001 -m 0.1
"""
#============================================================================#
# Imports and vars
#============================================================================#
import numpy as np
import random, argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#============================================================================#
# Class 
#============================================================================#
class Softmax:
    """ Softmax classifier implementation. 
        Args:
        - epochs: Number of iterations over complete training data
        - learningRate: A step size or a learning rate
        - batchSize: A mini-batch size(less than total number of training data)
        - regStrength: A regularization strength
        - momentum: A momentum value
    """
    __slots__ = ("epochs", "learningRate", "batchSize", "regStrength", "wt", "momentum", "velocity")
    
    def __init__(self, epochs, learningRate, batchSize, regStrength, momentum):
        self.epochs = epochs
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.regStrength = regStrength
        self.momentum = momentum
        self.velocity = None
        self.wt = None

    def train(self, xTrain, yTrain, xTest, yTest):
        """ Train a softmax classifier model on training data using stochastic gradient descent with mini-batches\\
        and momentum to minimize softmax (cross-entropy) loss of this single layer neural network.\\ 
        Calcualte the mean per-class accuracy for the training/testing data and the loss.
        
        Parameters: 
            - xTrain: Training input data
            - yTrain: Training labels
            - xTest: Testing input data
            - yTest: Testing labels
        
        Returns: Training Testing losses and Accuracy
        """
        # Get dimensionality
        D = xTrain.shape[1]  
        label = np.unique(yTrain)
        # Number of classes
        numOfClasses = len(label) 
        yTrainEnc = self.oneHotEncoding(yTrain, numOfClasses)
        yTestEnc = self.oneHotEncoding(yTest, numOfClasses)
        self.wt = 0.001 * np.random.rand(D, numOfClasses)
        self.velocity = np.zeros(self.wt.shape)
        trainLosses = []
        testLosses = []
        trainAcc = []
        testAcc = []
        
        for e in range(self.epochs):
            trainLoss = self.SGDWithMomentum(xTrain, yTrainEnc)
            testLoss, dw = self.computeLoss(xTest, yTestEnc)
            trainAcc.append(self.meanAccuracy(xTrain, yTrain))
            testAcc.append(self.meanAccuracy(xTest, yTest))
            trainLosses.append(trainLoss)
            testLosses.append(testLoss)
            print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainAcc : {:.7f}\t|\tTestAcc: {:.7f}".format(e, trainLoss, testLoss, trainAcc[-1], testAcc[-1]))
        return trainLosses, testLosses, trainAcc, testAcc

    def SGDWithMomentum(self, x, y):
        """ Stochastic gradient descent with mini-batches.\\
        Divide training data into mini-batches and compute loss and grad on that mini-batches and updates the weights.
        
        Parameters:
            - Input samples
            - Input labels
        
        Returns: Total loss computed
        """
        losses = []

        randomIndices = random.sample(range(x.shape[0]), x.shape[0])
        x = x[randomIndices]
        y = y[randomIndices]
        for i in range(0, x.shape[0], self.batchSize):
            Xbatch = x[i:i+self.batchSize]
            ybatch = y[i:i+self.batchSize]
            loss, dw = self.computeLoss(Xbatch, ybatch)
            self.velocity = (self.momentum * self.velocity) + (self.learningRate * dw)
            self.wt -= self.velocity
            losses.append(loss)
        return np.sum(losses) / len(losses)

    def softmaxEquation(self, scores):
        """ Calculate a softmax probability
        
        Parameters:
            - scores: matrix(wt * input sample)
        
        Returns: 
            Softmax probability
        """
        scores -= np.max(scores)
        prob = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
        return prob

    def computeLoss(self, x, yMatrix):
        """ Calculate the cross-entropy loss with regularization loss and gradient to update the weights.
        
        Parameters:
            - Input samples
            - yMatrix: Label as one-hot encoding
        
        Returns:
            Total loss and gradient
        """
        numOfSamples = x.shape[0]
        scores = np.dot(x, self.wt)
        prob = self.softmaxEquation(scores)

        loss = -np.log(np.max(prob)) * yMatrix
        regLoss = (1/2)*self.regStrength*np.sum(self.wt*self.wt)
        totalLoss = (np.sum(loss) / numOfSamples) + regLoss
        grad = ((-1 / numOfSamples) * np.dot(x.T, (yMatrix - prob))) + (self.regStrength * self.wt)
        return totalLoss, grad

    def meanAccuracy(self, x, y):
        """ Calculate mean-per class accuracy. """
        predY = self.predict(x)
        predY = predY.reshape((-1, 1))  # convert to column vector
        return np.mean(np.equal(y, predY))

    def predict(self, x):
        """ Predict the label based on input sample and a model. """
        return np.argmax(x.dot(self.wt), 1)

    def oneHotEncoding(self, y, numOfClasses):
        """ Convert a vector into one-hot encoding matrix where that particular column value is 1 and rest 0 for that row.
        
        Parameters:
            - Label vector
            - Number of unique labels
        
        Returns: 
            One-hot encoded matrix
        """
        y = np.asarray(y, dtype='int32')
        if len(y) > 1:
            y = y.reshape(-1)
        if not numOfClasses:
            numOfClasses = np.max(y) + 1
        yMatrix = np.zeros((len(y), numOfClasses))
        yMatrix[np.arange(len(y)), y] = 1
        return yMatrix

#============================================================================#
# Methods 
#============================================================================#
def plotGraph(trainLosses, testLosses, trainAcc, testAcc):
    """ Display the graph: Epochs vs. Cross Entropy Loss 
    
    Parameters:
        - trainLosses: List of training loss over every epochs
        - testLosses: List of testing loss over every epochs
        - trainAcc: List of training accuracy over every epochs
        - testAcc: List of testing accuracy over every epochs
    """
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label="Train loss")
    plt.plot(testLosses, label="Test loss")
    plt.legend(loc='best')
    plt.title("Epochs vs. Cross Entropy Loss")
    plt.xlabel("Number of Iteration or Epochs")
    plt.ylabel("Cross Entropy Loss")

    plt.subplot(1, 2, 2)
    plt.plot(trainAcc, label="Train Accuracy")
    plt.plot(testAcc, label="Test Accuracy")
    plt.legend(loc='best')
    plt.title("Epochs vs. Mean per class Accuracy")
    plt.xlabel("Number of Iteration or Epochs")
    plt.ylabel("Mean per class Accuracy")
    
    plt.subplots_adjust(wspace=0.5)

    plt.show()

def plotDecisionBoundary(x, y):
    """ Display the decision boundary to display a sample with region. """
    markers = ('+', '.', 'x')
    colors = ('blue', 'dimgrey', 'maroon')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    xx, yy = makeMeshGrid(x, y)
    plotContours(plt, sm, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    # Plot also the training points
    for idx, cl in enumerate(np.unique(y)):
        xBasedOnLabel = x[np.where(y[:,0] == cl)]
        plt.scatter(x=xBasedOnLabel[:, 0], y=xBasedOnLabel[:, 1], c=cmap(idx),
                    cmap=plt.cm.coolwarm, marker=markers[idx], label=cl)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Feature X1")
    plt.ylabel("Feature X2")
    plt.title("Softmax Classifier on Iris Dataset(Decision Boundary)")
    plt.xticks()
    plt.yticks()
    plt.legend(loc='upper left')
    plt.show()

def readData(filename):
    """ Read data from file and divide into input sample and a label. """
    dataMatrix = np.loadtxt(filename)
    np.random.shuffle(dataMatrix)
    X = dataMatrix[:, 1:]
    y = dataMatrix[:, 0].astype(int)
    y = y.reshape((-1, 1))
    y -= 1
    return X, y

#============================================================================#
# Main 
#============================================================================#
TRAIN_FILENAME = "./data/iris-train.txt"
TEST_FILENAME = "./data/iris-test.txt"

if __name__=='__main__':
    trainX, trainY = readData(TRAIN_FILENAME)   
    testX, testY = readData(TEST_FILENAME)

    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", dest="epochs", default=1000,
                        type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learningrate", dest="learningRate", default=0.07,
                        type=float, help="Learning rate or step size")
    parser.add_argument("-bs", "--batchSize", dest="batchSize", default=10,
                        type=int, help="Number of sample in mini-batches")
    parser.add_argument("-r", "--regStrength", dest="regStrength", default=0.001,
                        type=float, help="L2 weight decay regularization lambda value")
    parser.add_argument("-m", "--momentum", dest="momentum", default=0.05,
                        type=float, help="A momentum value")

    args = parser.parse_args()

    print("Epochs: {} | Learning Rate: {} | Batch Size: {} | Regularization Strength: {} | " "Momentum: {} |".format(
            args.epochs,
            args.learningRate,
            args.batchSize,
            args.regStrength,
            args.momentum
        ))
    
    epochs = int(args.epochs)
    learningRate = float(args.learningRate)
    batchSize = int(args.batchSize)
    regStrength = int(args.regStrength)
    momentum = int(args.momentum)
    
    ### Train
    sm = Softmax(epochs=epochs, learningRate=learningRate, batchSize=batchSize,
                    regStrength=regStrength, momentum=momentum)
    trainLosses, testLosses, trainAcc, testAcc = sm.train(trainX, trainY, testX, testY) 

    plotGraph(trainLosses, testLosses, trainAcc, testAcc)
    plotDecisionBoundary(trainX, trainY)
    plotDecisionBoundary(testX, testY)