import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import logging

import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, train_inputs, train_outputs):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs.reshape(-1, 1)
        self.weights = np.zeros((train_inputs.shape[1], 1))
        self.bias = 0
        self.loss = []

    def cost_func(self, prediction, train_output):
        return np.mean((prediction - train_output) ** 2)

    def sgd(self, prediction, train_output):
        prediction = prediction.reshape(-1, 1)
        train_output = train_output.reshape(-1, 1)
        gradient_weights = np.dot(self.train_inputs.T, (prediction - train_output)) / self.train_inputs.shape[0]
        gradient_bias = np.mean(prediction - train_output)
        return gradient_weights, gradient_bias

    def forward_propagation(self):
        return np.dot(self.train_inputs, self.weights) + self.bias

    def update_weights(self, prediction, train_output, learning_rate):
        gradient_weights, gradient_bias = self.sgd(prediction, train_output)
        self.weights -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_bias

    def train(self, learning_rate, iters):
        if self.train_inputs.shape[1] == 1:
            fig, ax = plt.subplots()
            ax.scatter(self.train_inputs, self.train_outputs, label='Training data')
            line, = ax.plot(self.train_inputs, self.forward_propagation(), color='red', label='Fit')

            def update(i):
                prediction = self.forward_propagation()
                cost = self.cost_func(prediction, self.train_outputs)
                self.update_weights(prediction, self.train_outputs, learning_rate)
                self.loss.append(cost)
                # print(f"Iteration {i}: w={self.weights.flatten()}, b={self.bias}, cost={cost}")
                line.set_ydata(self.forward_propagation())
                return [line]

            ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
            ani.save('linear_regression_A.gif', writer='ffmpeg')

            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.title('Linear Regression')
            plt.legend()
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.set_xlim(0, iters)
            ax.set_ylim(0, 1.5 * self.cost_func(self.forward_propagation(), self.train_outputs))
            line, = ax.plot([], [], color='red', label='Cost')

            def update(i):
                prediction = self.forward_propagation()
                cost = self.cost_func(prediction, self.train_outputs)
                self.update_weights(prediction, self.train_outputs, learning_rate)
                self.loss.append(cost)
                # print(f"Iteration {i}: w={self.weights.flatten()}, b={self.bias}, cost={cost}")
                if i % 10 == 0:
                    print(f"Iteration {i}, cost={cost}")    
                line.set_data(range(len(self.loss)), self.loss)
                return [line]

            ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
            ani.save('linear_regression_cost.gif', writer='ffmpeg')

            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Cost Function Convergence')
            plt.legend()
            plt.show()

        return self.weights, self.bias, self.loss

    def predict(self, test_input):
        return np.dot(test_input, self.weights) + self.bias

    def accuracy(self, test_input, test_output):
        prediction = self.predict(test_input)
        return 1 - self.cost_func(prediction, test_output) / np.mean(test_output ** 2)

# data = pd.read_csv('../datasets/Student_Performance.csv')

# train_size = 500
# test_size = 199

# data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
# train_input = np.array(data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']][:train_size]).reshape(train_size, 5)
# train_output = np.array(data['Performance Index'][:train_size]).reshape(train_size, 1)

# print("Train input shape:", train_input.shape)
# print("Train output shape:", train_output.shape)

# test_input = np.array(data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']][train_size:train_size+test_size]).reshape(test_size, 5)
# test_output = np.array(data['Performance Index'][train_size:train_size+test_size]).reshape(test_size, 1)

# print("Test input shape:", test_input.shape)
# print("Test output shape:", test_output.shape)

# model = LinearRegression(train_input, train_output)
# parameters, bias, loss = model.train(learning_rate=0.0001, iters=50)

# accuracy = model.accuracy(test_input, test_output)



# print(f'Accuracy: {accuracy}')