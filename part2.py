## CS 6375 Assignment 2
## Christopher Chan (ccc180002) and Thomas Pham (TTP190005)
## Late days used: 2

## Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

## Implementation of Neural Network 
class NeuralNetwork:
  
  ## Fetch data
  def fetchData(self):
    df = pd.read_csv(FILE_PATH)
    return df
  
  ## Preprocess data
  def preProcessData(self, df):
    df_normalized = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    df['Species'] = df['Species'].astype('category').cat.codes
    df_normalized['Species'] = df['Species']
    return df_normalized    
  
  ## Create train and Test split implementation
  def createTrainTestSplit(self, df, splitRatio):
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = df[['Species']]
    targets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    Y = np.array([targets[int(x)] for x in df['Species']])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = splitRatio, random_state = 42)

    return X_train, X_test, Y_train, Y_test
  
  ## Constructor for Neural Network Class
  def __init__(self, _inputSize, _hiddenSize, _outputSize):

    self.inputSize = _inputSize
    self.hiddenSize = _hiddenSize
    self.outputSize = _outputSize

    np.random.seed(2)
    self.w1 = np.random.randn(self.inputSize, self.hiddenSize) 
    self.b1 = np.random.randn(1, self.hiddenSize) 
    self.w2 = np.random.randn(self.hiddenSize, self.outputSize) 
    self.b2 = np.random.randn(1, self.outputSize)
    
    self.momentum = {
      'w1': np.zeros_like(self.w1),
      'w2': np.zeros_like(self.w2),
      'b1': np.zeros_like(self.b1),
      'b2': np.zeros_like(self.b2)
    }

  ## Function to reset weights, bias, and momentum of neural network to be reused
  def resetParameters(self):
    np.random.seed(2)
    self.w1 = np.random.randn(self.inputSize, self.hiddenSize) 
    self.b1 = np.random.randn(1, self.hiddenSize) 
    self.w2 = np.random.randn(self.hiddenSize, self.outputSize) 
    self.b2 = np.random.randn(1, self.outputSize)
    
    self.momentum = {
      'w1': np.zeros_like(self.w1),
      'w2': np.zeros_like(self.w2),
      'b1': np.zeros_like(self.b1),
      'b2': np.zeros_like(self.b2)
    }

  ## Activation Functions for forward pass
  def activation_function_forward(self, activation_function_name, X):
    if activation_function_name == 'sigmoid':
      return 1 / (1 + np.exp(-X))
    elif activation_function_name == 'tanh':
      return np.tanh(X)
    elif activation_function_name == 'relu':
      return np.maximum(0,X)
  
  ## Derivative of Activation Functions for backward pass
  def activation_function_backward(self, activation_function_name, X):
    if activation_function_name == 'sigmoid':
      return X * (1 - X)
    elif activation_function_name == 'tanh':
      return 1 - np.tanh(X) ** 2
    elif activation_function_name == 'relu':
      temp = np.where(X < 0, 0, X)
      temp = np.where(X >= 0, 1, X)
      return temp
  
  ## Forward Pass Implementation
  def forward_pass(self, X, w1, w2, b1, b2, activation_function_name):
    layer1 = self.activation_function_forward(activation_function_name, np.dot(X, w1) + b1)
    layer2 = self.activation_function_forward(activation_function_name, np.dot(layer1, w2) + b2)
    return layer1, layer2
  
  ## Backward Pass Implementation
  def backward_pass(self, X, Y, old_w1, old_w2, old_b1, old_b2, layer1, layer2, activation_function_name, learning_rate, optimizer, momentum_rate):
    ## Backward Pass without Momentum Optimization
    if optimizer == 'No':
      layer2_delta = (Y - layer2) * self.activation_function_backward(activation_function_name, layer2)
      new_w2 = old_w2 + layer1.T.dot(layer2_delta) * learning_rate
      new_b2 = old_b2 + np.sum(layer2_delta, axis=0, keepdims=True) * learning_rate

      layer1_delta = (layer2_delta.dot(new_w2.T)) * self.activation_function_backward(activation_function_name, layer1)
      new_w1 = old_w1 + (X.values).T.dot(layer1_delta) * learning_rate
      new_b1 = old_b1 + np.sum(layer1_delta, axis=0, keepdims=True) * learning_rate

      return new_w1, new_w2, new_b1, new_b2
    ## Backward Pass with Momentum Optimization
    elif optimizer == 'Yes':
      layer2_delta = (Y - layer2) * self.activation_function_backward(activation_function_name, layer2)
      new_w2 = old_w2 + layer1.T.dot(layer2_delta) * learning_rate + self.momentum['w2'] * momentum_rate
      new_b2 = old_b2 + np.sum(layer2_delta, axis=0, keepdims=True) * learning_rate + self.momentum['b2'] * momentum_rate

      layer1_delta = (layer2_delta.dot(new_w2.T)) * self.activation_function_backward(activation_function_name, layer1)
      new_w1 = old_w1 + (X.values).T.dot(layer1_delta) * learning_rate + self.momentum['w1'] * momentum_rate
      new_b1 = old_b1 + np.sum(layer1_delta, axis=0, keepdims=True) * learning_rate + self.momentum['b1'] * momentum_rate

      ## Update momentum 
      new_momentum = {
          'w1': new_w1 - old_w1,
          'w2': new_w2 - old_w2,
          'b1': new_b1 - old_b1,
          'b2': new_b2 - old_b2
      }
      return new_w1, new_w2, new_b1, new_b2, new_momentum
    
  
  ## Implementation of Training Neural Network
  def train_neural_network(self, X, Y, activation_function_name, learning_rate, epochs, optimizer):
    ## Array to hold the error history
    misclasifficationHistory = pd.DataFrame(columns = ['iteration', 'missclassification'])
    misclasiffications = 0
    array_index = 0
    for epoch in range(epochs):
      layer1, layer2 = self.forward_pass(X, self.w1, self.w2, self.b1, self.b2, activation_function_name)

      if (optimizer == 'No'):
        self.w1, self.w2, self.b1, self.b2 = self.backward_pass(X, Y, self.w1, self.w2, self.b1, self.b2, layer1, layer2, activation_function_name, learning_rate, optimizer , MOMENTUM_RATE)
      else:
        self.w1, self.w2, self.b1, self.b2, self.momentum = self.backward_pass(X, Y, self.w1, self.w2, self.b1, self.b2, layer1, layer2, activation_function_name, learning_rate, optimizer , MOMENTUM_RATE)
      
      ## Calculates the misclassification percent per iteration and append to error history
      misclasiffications = 0
      predIndices = np.argmax(layer2, axis=1)
      predictions = np.zeros_like(layer2)
      predictions[np.arange(layer2.shape[0]),predIndices] = 1
      for pred, actual in zip(predictions, Y):
          if not np.array_equal(pred, actual):
            misclasiffications += 1

      missPercentage = misclasiffications/ (len(Y))
      misclasifficationHistory.loc[array_index] = [epoch, missPercentage]
      array_index += 1
      misclasiffications = 0
  
    return misclasifficationHistory
  
  ## Implementation of Function to Get Predictions and Compare to Testing Data after Neural Network has been trained
  def test_neural_network(self, X, Y, activation_function_name, n):

    for datapoint in range(n):
      layer1, layer2 = self.forward_pass(X, self.w1, self.w2, self.b1, self.b2, activation_function_name)
      misclasiffications = 0
      predIndices = np.argmax(layer2, axis=1)
      predictions = np.zeros_like(layer2)
      predictions[np.arange(layer2.shape[0]),predIndices] = 1
      for pred, actual in zip(predictions, Y):
          if not np.array_equal(pred, actual):
            misclasiffications += 1

      missPercentage = misclasiffications/ (len(Y))
    return missPercentage
  
  ## Implementation of Function to Display Graph of Percent Error VS Iterations
  def displayTrainingHistory(self, history, activation_function_name, alpha_value, optimization):
    if optimization == 'No':
      plt.plot(history['iteration'], history['missclassification'], label = alpha_value)
      plt.ylabel('Percent Error')
      plt.xlabel('Number of Iterations')
      plt.title('Percent Error vs Iterations for ' + activation_function_name + ", without optimization" + ", alpha: " + str(alpha_value))
      plt.show()
    else:
      plt.plot(history['iteration'], history['missclassification'], label = alpha_value)
      plt.ylabel('Percent Error')
      plt.xlabel('Number of Iterations')
      plt.title('Percent Error vs Iterations for ' + activation_function_name + ", with optimization" +  ", alpha: " + str(alpha_value))
      plt.show()

  ## Implementation of Function to Display Correlation Matrix
  def displayCorrelationMatrix(self):
    df = pd.read_csv(FILE_PATH)
    corr = df.corr()
    sns.heatmap(corr,cmap=sns.diverging_palette(220, 10),annot=True,fmt='.2f',)
    plt.title('Correlation Matrix')
    plt.show()

## Implementation of Class to Compare Different Activation Functions and Learning Rates
class CompareNetworks:
  ## Implementation of Function to Compare Learing Rates
  def compareLearningRates(self, activation_function_name, alpha1, alpha2, alpha3, iterations):
    ANN = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    df = ANN.fetchData()
    df_normalized = ANN.preProcessData(df)  
    X_train, X_test, Y_train, Y_test = ANN.createTrainTestSplit(df_normalized, SPLIT_RATIO)

    trainhistory_1 = ANN.train_neural_network(X_train, Y_train, activation_function_name , alpha1, EPOCHS, OPTIMIZATION)
    train_error_1 = trainhistory_1.loc[iterations-1]['missclassification']
    test_error_1 = ANN.test_neural_network(X_test, Y_test, activation_function_name, len(Y_test))
    ANN.resetParameters()

    trainhistory_2 = ANN.train_neural_network(X_train, Y_train, activation_function_name , alpha2, EPOCHS, OPTIMIZATION)
    train_error_2 = trainhistory_2.loc[iterations-1]['missclassification']
    test_error_2 = ANN.test_neural_network(X_test, Y_test, activation_function_name, len(Y_test))
    ANN.resetParameters()

    trainhistory_3 = ANN.train_neural_network(X_train, Y_train, activation_function_name , alpha3, EPOCHS, OPTIMIZATION)
    train_error_3 = trainhistory_3.loc[iterations-1]['missclassification']
    test_error_3 = ANN.test_neural_network(X_test, Y_test, activation_function_name, len(Y_test))

    return train_error_1, test_error_1, train_error_2, test_error_2, train_error_3, test_error_3, trainhistory_1, trainhistory_2, trainhistory_3
  
  ## Implementation of Function to Display Error History of Various Learning Rates 
  def displayCompareLearningRates(self, history1, history2, history3, a1, a2, a3, activation_function_name):
    plt.plot(history1['iteration'], history1['missclassification'], label = "Alpha: " + str(a1))
    plt.plot(history2['iteration'], history2['missclassification'], label = "Alpha: " + str(a2))
    plt.plot(history3['iteration'], history3['missclassification'], label = "Alpha: " + str(a3))
    plt.legend()
    plt.ylabel('Percent Error')
    plt.xlabel('Number of Iterations')
    plt.title("Percent Error VS Iterations for Training Data for " + activation_function_name)
    plt.show()
  
  ## Prints Out Summary of Results
  def showComparisonLearningRatesResults(self, train_error_1, test_error_1, train_error_2, test_error_2, train_error_3, test_error_3, a1, a2, a3, activation_function_name):
    print(activation_function_name + " activation function with" + " learning rate: " + str(a1) + ", " + "the train error was: " + str(train_error_1) + " and the test error was: " + str(test_error_1) + '\n')
    print(activation_function_name + " activation function with" + " learning rate: " + str(a2) + ", " + "the train error was: " + str(train_error_2) + " and the test error was: " + str(test_error_2) + '\n')
    print(activation_function_name + " activation function with" + " learning rate: " + str(a3) + ", " + "the train error was: " + str(train_error_3) + " and the test error was: " + str(test_error_3) + '\n')

  ## Implementation of Function to Compare Activation Functions
  def compareActivationFunctions(self, alpha):
    ANN = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    df = ANN.fetchData()
    df_normalized = ANN.preProcessData(df)  
    X_train, X_test, Y_train, Y_test = ANN.createTrainTestSplit(df_normalized, SPLIT_RATIO)
    
    train_history_sigmoid = ANN.train_neural_network(X_train, Y_train, 'sigmoid' , alpha, EPOCHS, OPTIMIZATION)
    train_error_sigmoid = train_history_sigmoid.loc[EPOCHS-1]['missclassification']
    test_error_sigmoid = ANN.test_neural_network(X_test, Y_test, 'sigmoid', len(Y_test))
    ANN.resetParameters()

    train_history_tanh = ANN.train_neural_network(X_train, Y_train, 'tanh' , alpha, EPOCHS, OPTIMIZATION)
    train_error_tanh = train_history_tanh.loc[EPOCHS-1]['missclassification']
    test_error_tanh = ANN.test_neural_network(X_test, Y_test,'tanh', len(Y_test))
    ANN.resetParameters()

    train_history_relu = ANN.train_neural_network(X_train, Y_train, 'relu' , alpha, EPOCHS, OPTIMIZATION)
    train_error_relu = train_history_relu.loc[EPOCHS-1]['missclassification']
    test_error_relu = ANN.test_neural_network(X_test, Y_test, 'relu', len(Y_test))

    return train_error_sigmoid, test_error_sigmoid, train_error_tanh, test_error_tanh, train_error_relu, test_error_relu, train_history_sigmoid, train_history_tanh, train_history_relu
  

def displayCompareActivationFunctions(self, train_history_sigmoid, train_history_tanh, train_history_relu):
  plt.plot(train_history_sigmoid['iteration'], train_history_sigmoid['missclassification'], label = 'sigmoid')
  plt.plot(train_history_tanh['iteration'], train_history_tanh['missclassification'], label = 'tanh')
  plt.plot(train_history_relu['iteration'], train_history_relu['missclassification'], label = 'relu')
  plt.legend()
  plt.ylabel('Percent Error')
  plt.xlabel('Number of Iterations')
  plt.title("Percent Error VS Iterations for Training Data for activation functions")
  plt.show()

  

## Parameters
## Options for ACTIV_FUNC: sigmoid, tanh, relu
## Options for OPTMIZATION: Yes, No

FILE_PATH = "https://raw.githubusercontent.com/thomas944/ml-assignment-2/main/Iris.csv"
MOMENTUM_RATE = 0.9
LEARNING_RATE = 0.002
EPOCHS = 10000
SPLIT_RATIO = 0.2
ACTIV_FUNC = 'relu'
OPTIMIZATION = 'Yes'
INPUT_SIZE = 4
HIDDEN_SIZE = 5
OUTPUT_SIZE = 3


## Variables for Comparison
A1 = 0.2
A2 = 0.5
A3 = 0.7

def main():
  # myANN = NeuralNetwork(4,5,3)
  # df = myANN.fetchData()
  # df_normalized = myANN.preProcessData(df)
  # X_train, X_test, Y_train, Y_test = myANN.createTrainTestSplit(df_normalized, SPLIT_RATIO)
  # trainHistory = myANN.train_neural_network(X_train, Y_train, ACTIV_FUNC , LEARNING_RATE, EPOCHS, OPTIMIZATION)
  # myANN.test_neural_network(X_test, Y_test, ACTIV_FUNC, len(Y_test))
  # myANN.displayTrainingHistory(trainHistory, ACTIV_FUNC, LEARNING_RATE, OPTIMIZATION)


  # Uncomment this section to see the comparisons ##
  # compare = CompareNetworks()
  # train_error_1, test_error_1, train_error_2, test_error_2, train_error_3, test_error_3, trainhistory_1, trainhistory_2, trainhistory_3 = compare.compareLearningRates('sigmoid',A1, A2, A3, EPOCHS)
  # compare.displayCompareLearningRates(trainhistory_1, trainhistory_2, trainhistory_3, A1, A2, A3, 'sigmoid')
  # compare.showComparisonLearningRatesResults(train_error_1, test_error_1, train_error_2, test_error_2, train_error_3, test_error_3, A1, A2, A3, 'sigmoid')
  #myANN.displayCorrelationMatrix()

  compare = CompareNetworks()
  train_error_sigmoid, test_error_sigmoid, train_error_tanh, test_error_tanh, train_error_relu, test_error_relu, train_history_sigmoid, train_history_tanh, train_history_relu = compare.compareActivationFunctions(0.002)
  plt.plot(train_history_sigmoid['iteration'], train_history_sigmoid['missclassification'], label = 'sigmoid')
  plt.plot(train_history_tanh['iteration'], train_history_tanh['missclassification'], label = 'tanh')
  plt.plot(train_history_relu['iteration'], train_history_relu['missclassification'], label = 'relu')
  plt.legend()
  plt.ylabel('Percent Error')
  plt.xlabel('Number of Iterations')
  plt.title("Percent Error VS Iterations for Training Data for activation functions")
  plt.show()
  print(train_error_sigmoid, test_error_sigmoid, train_error_tanh, test_error_tanh, train_error_relu, test_error_relu)
if __name__=="__main__":
  main()