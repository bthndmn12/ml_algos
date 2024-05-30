import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import logging

class DataLoader():
    def load_data(self, url):
        
        data = pd.read_csv(url)
        data = data.dropna()
        return data

    def split_data(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

class PolyRegression():
    def __init__(self, train_inputs, train_outputs, degree):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.degree = degree
        self.weights = np.zeros((self.degree + 1, 1))
        self.bias = 0
        self.loss = []

    def cost_func(self, prediction, train_output):
        return np.mean((prediction - train_output) ** 2)

    # def transform_inputs(self, inputs):
    #     transformed_inputs = inputs
    #     for i in range(2, self.degree + 1):
    #         transformed_inputs = np.concatenate((transformed_inputs, inputs ** i), axis=1)
    #     transformed_inputs = np.concatenate((np.ones((transformed_inputs.shape[0], 1)), transformed_inputs), axis=1)
    #     return transformed_inputs

    # used transform and normalize function from here https://www.geeksforgeeks.org/polynomial-regression-from-scratch-using-python/?ref=lbp
    def transform_inputs(self, X):
        # initialize X_transform
        X_transform = np.ones((X.shape[0], 1))

        for j in range(1, self.degree + 1):
            x_pow = np.power(X, j)
            # append x_pow to X_transform
            X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)

        return X_transform
    
    # def normalize_inputs(self, inputs):
    #     # Normalize the inputs
    #     mean = np.mean(inputs, axis=0)
    #     std = np.std(inputs, axis=0)
    #     # Avoid division by zero by setting std to 1 where std is 0
    #     std[std == 0] = 1
    #     return (inputs - mean) / std
    def normalize_inputs( self, X ) :
         
        X[:, 1:] = ( X[:, 1:] - np.mean( X[:, 1:], axis = 0 ) ) / np.std( X[:, 1:], axis = 0 )
         
        return X

    def forward_propagation(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def update_weights(self, prediction, train_output, normalized_inputs, learning_rate):
        prediction = prediction.reshape(-1, 1)
        train_output = train_output.reshape(-1, 1)

        # transformed_inputs = self.transform_inputs(self.train_inputs)
        # normalized_inputs = self.normalize_inputs(transformed_inputs)

        gradient_weights = np.dot(normalized_inputs.T, (prediction - train_output)) / normalized_inputs.shape[0]
        gradient_bias = np.mean(prediction - train_output)

        self.weights -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_bias

    def train(self, learning_rate, iters):
        fig, ax = plt.subplots()
        ax.scatter(self.train_inputs, self.train_outputs, label='Training data')

        transformed_inputs = self.transform_inputs(self.train_inputs)
        normalized_inputs = self.normalize_inputs(transformed_inputs)
        line, = ax.plot(self.train_inputs, self.forward_propagation(normalized_inputs), color='red', label='Fit')
        logging.basicConfig(level=logging.INFO)
        def update(i):
            prediction = self.forward_propagation(normalized_inputs)
            cost = self.cost_func(prediction, self.train_outputs)
            self.update_weights(prediction, self.train_outputs, normalized_inputs, learning_rate)
            self.loss.append(cost)
            line.set_ydata(self.forward_propagation(normalized_inputs))
            # print("epoch, w, b, c: ", i, self.weights, self.bias, cost)
            
            logging.info(f"Iteration {i}, cost={cost}")
            # print(f"Iteration {i}, cost={cost}")
            return line,

        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
        ani.save('linear_regression_20.gif', writer='ffmpeg')

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Polynomial Regression')
        plt.legend()
        plt.show()

        return self.weights, self.bias, self.loss

    def predict(self, test_input):
        test_input = test_input.reshape(-1, 1)
        transformed_inputs = self.transform_inputs(test_input)
        normalized_inputs = self.normalize_inputs(transformed_inputs)
        return np.dot(normalized_inputs, self.weights) + self.bias

    def accuracy(self, test_input, test_output):
        prediction = self.predict(test_input)
        return 1 - self.cost_func(prediction, test_output) / np.mean(test_output ** 2)

# # data = SyntheticData(SyntheticData.DataType.SINE, 2000, 0.1)
# data = SyntheticData(SyntheticData.DataType.LOGARITHMIC, 2000, 0.1,10)
# sin_data = data.data


# # data_loader = DataLoader()
# # data = data_loader.load_data("../datasets/data_for_lr.csv")
# print(data.data.head())


# train_input = np.array(sin_data['x'][0:1500]).reshape(1500, 1)
# train_output = np.array(sin_data['y'][0:1500]).reshape(1500, 1)

# test_input = np.array(sin_data['x'][1500:2000]).reshape(500, 1)
# test_output = np.array(sin_data['y'][1500:2000]).reshape(500, 1)

# model = PolyRegression(train_input, train_output,8)
# parameters, bias, loss = model.train(learning_rate=0.01, iters=4000)


# print("Weights:", parameters)
# print("Bias:", bias)
# print("Loss:", loss)
# test_accuracy = model.accuracy(test_input, test_output)
# print("Test Accuracy:", test_accuracy)
