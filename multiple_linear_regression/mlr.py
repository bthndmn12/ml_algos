import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

class LinearRegression:
    def __init__(self, train_inputs, train_outputs):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs.reshape(-1, 1)  # Ensuring outputs are a column vector
        self.weights = np.zeros((train_inputs.shape[1], 1))
        self.bias = 0
        self.loss = []

    def cost_func(self, prediction, train_output):
        # Calculate the cost
        return np.mean((prediction - train_output) ** 2)

    # stochastic gradient descent for linear regression    
    def sgd(self, prediction, train_output):
        
        prediction = prediction.reshape(-1, 1)
        train_output = train_output.reshape(-1, 1)
        
        # Calculate the gradients
        gradient_weights = np.dot(self.train_inputs.T, (prediction - train_output)) / self.train_inputs.shape[0]
        gradient_bias = np.mean(prediction - train_output)

        return gradient_weights, gradient_bias



    def forward_propagation(self):
        # Forward propagation
        return np.dot(self.train_inputs, self.weights) + self.bias

    def update_weights(self, prediction, train_output, learning_rate):
        # Reshape prediction and train_output
        
        gradient_weights, gradient_bias = self.sgd(prediction, train_output)

        # Update weights and bias
        self.weights -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_bias

    def train(self, learning_rate, iters):
        if self.train_inputs.shape[1] == 1:
            # Set up the plot for single feature
            fig, ax = plt.subplots()
            ax.scatter(self.train_inputs, self.train_outputs, label='Training data')
            line, = ax.plot(self.train_inputs, self.forward_propagation(), color='red', label='Fit')

            def update(i):
                prediction = self.forward_propagation()
                cost = self.cost_func(prediction, self.train_outputs)
                self.update_weights(prediction, self.train_outputs, learning_rate)
                self.loss.append(cost)

                # Log weights, bias, and cost
                print(f"Iteration {i}: w={self.weights.flatten()}, b={self.bias}, cost={cost}")

                # Update the line in the plot
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
            # Set up the plot for cost function over iterations
            fig, ax = plt.subplots()
            ax.set_xlim(0, iters)
            ax.set_ylim(0, 1.5 * self.cost_func(self.forward_propagation(), self.train_outputs))
            line, = ax.plot([], [], color='red', label='Cost')

            def update(i):
                prediction = self.forward_propagation()
                cost = self.cost_func(prediction, self.train_outputs)
                self.update_weights(prediction, self.train_outputs, learning_rate)
                self.loss.append(cost)

                # Log weights, bias, and cost
                print(f"Iteration {i}: w={self.weights.flatten()}, b={self.bias}, cost={cost}")

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
        # Predict the output
        return np.dot(test_input, self.weights) + self.bias

    def accuracy(self, test_input, test_output):
        # Calculate the accuracy
        prediction = self.predict(test_input)
        return 1 - self.cost_func(prediction, test_output) / np.mean(test_output ** 2)
# class LinearRegression:
#     def __init__(self, train_inputs, train_outputs):
#         self.train_inputs = train_inputs
#         self.train_outputs = train_outputs
#         self.weights = np.zeros((train_inputs.shape[1], 1))
#         self.bias = 0
#         self.loss = []

#     def cost_func(self, prediction, train_output):
#         # Calculate the cost
#         return np.mean((prediction - train_output) ** 2)

#     def forward_propagation(self):
#         # Forward propagation
#         return np.dot(self.train_inputs, self.weights) + self.bias

#     def update_weights(self, prediction, train_output, learning_rate):
#         # Reshape prediction and train_output
#         prediction = prediction.reshape(-1, 1)
#         train_output = train_output.reshape(-1, 1)

#         # Calculate the gradients
#         gradient_weights = np.dot(self.train_inputs.T, (prediction - train_output)) / self.train_inputs.shape[0]
#         gradient_bias = np.mean(prediction - train_output)

#         # Update weights and bias
#         self.weights -= learning_rate * gradient_weights
#         self.bias -= learning_rate * gradient_bias

#     def train(self, learning_rate, iters):
#         if self.train_inputs.shape[1] == 1:
#             # Set up the plot for single feature
#             fig, ax = plt.subplots()
#             ax.scatter(self.train_inputs, self.train_outputs, label='Training data')
#             line, = ax.plot(self.train_inputs, self.forward_propagation(), color='red', label='Fit')

#             def update(i):
#                 prediction = self.forward_propagation()
#                 cost = self.cost_func(prediction, self.train_outputs)
#                 self.update_weights(prediction, self.train_outputs, learning_rate)
#                 self.loss.append(cost)

#                 # Update the line in the plot
#                 line.set_ydata(self.forward_propagation())
#                 return [line]

#             ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
#             ani.save('linear_regression_A.gif', writer='ffmpeg')

#             plt.xlabel('Input')
#             plt.ylabel('Output')
#             plt.title('Linear Regression')
#             plt.legend()
#             plt.show()
#         else:
#             # Set up the plot for cost function over iterations
#             fig, ax = plt.subplots()
#             ax.set_xlim(0, iters)
#             ax.set_ylim(0, 1.5 * self.cost_func(self.forward_propagation(), self.train_outputs))
#             line, = ax.plot([], [], color='red', label='Cost')

#             def update(i):
#                 prediction = self.forward_propagation()
#                 cost = self.cost_func(prediction, self.train_outputs)
#                 self.update_weights(prediction, self.train_outputs, learning_rate)
#                 self.loss.append(cost)

#                 line.set_data(range(len(self.loss)), self.loss)
#                 return [line]

#             ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
#             ani.save('linear_regression_cost.gif', writer='ffmpeg')

#             plt.xlabel('Iteration')
#             plt.ylabel('Cost')
#             plt.title('Cost Function Convergence')
#             plt.legend()
#             plt.show()

#         return self.weights, self.bias, self.loss

#     def predict(self, test_input):
#         # Predict the output
#         return np.dot(test_input, self.weights) + self.bias

#     def accuracy(self, test_input, test_output):
#         # Calculate the accuracy
#         prediction = self.predict(test_input)
#         return 1 - self.cost_func(prediction, test_output) / np.mean(test_output ** 2)


# Load the dataset
data = pd.read_csv('../datasets/Student_Performance.csv')

# Split the dataset into training and testing sets
train_size = 500
test_size = 199

data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
# Training dataset and labels
train_input = np.array(data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']][:train_size]).reshape(train_size, 5)
train_output = np.array(data['Performance Index'][:train_size]).reshape(train_size, 1)

# Check the shapes of the training data
print("Train input shape:", train_input.shape)
print("Train output shape:", train_output.shape)

# Testing dataset and labels
test_input = np.array(data[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']][train_size:train_size+test_size]).reshape(test_size, 5)
test_output = np.array(data['Performance Index'][train_size:train_size+test_size]).reshape(test_size, 1)

# Check the shapes of the testing data
print("Test input shape:", test_input.shape)
print("Test output shape:", test_output.shape)

model = LinearRegression(train_input, train_output)
parameters, bias, loss = model.train(learning_rate=0.0001, iters=50)


# Evaluate the model
accuracy = model.accuracy(test_input, test_output)



print(f'Accuracy: {accuracy}')