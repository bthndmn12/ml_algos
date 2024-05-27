
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DataLoader():
    def load_data(self, url):
        # Load the dataset

        data = pd.read_csv(url)

        # Drop the missing values
        data = data.dropna()

        return data

    def split_data(self, data, train_ratio=0.8):
        # Split the data into training and testing sets
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]

        return train_data, test_data

class LinearRegression():
    def __init__(self, train_inputs, train_outputs):
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.weights = np.zeros((train_inputs.shape[1], 1))
        
        self.bias = 0
        self.loss = []

    def cost_func(self, prediction, train_output):
        # Calculate the cost
        return np.mean((prediction - train_output) ** 2)

    def forward_propagation(self):
        # Forward propagation
        return np.dot(self.train_inputs, self.weights) + self.bias

    def update_weights(self, prediction, train_output, learning_rate):
        # Update the weights
        print("train_inputs.T shape: ", self.train_inputs.T.shape)
        print("prediction shape: ", prediction.shape)
        print("train_output shape: ", train_output.shape)
        print("prediction-train_output shape: ", (prediction - train_output).shape)
        print("train_inputs shape: ", self.train_inputs.shape[0])

        
        prediction = prediction.reshape(-1, 1)
        train_output = train_output.reshape(-1, 1)

        # Calculate the gradients
        gradient_weights = np.dot(self.train_inputs.T, (prediction - train_output)) / self.train_inputs.shape[0]
        gradient_bias = np.mean(prediction - train_output)

        # Update weights and bias
        self.weights -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_bias

    def train(self, learning_rate, iters):
        fig, ax = plt.subplots()
        ax.scatter(self.train_inputs, self.train_outputs, label='Training data')
        line, = ax.plot(self.train_inputs, self.forward_propagation(), color='red', label='Fit')

        def update(i):
            prediction = self.forward_propagation()
            cost = self.cost_func(prediction, self.train_outputs)
            self.update_weights(prediction, self.train_outputs, learning_rate)
            print("w, b, c: ", self.weights, self.bias, cost)
            self.loss.append(cost)
            line.set_ydata(prediction)
            return line,

        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
        ani.save('linear_regression_A.gif', writer='ffmpeg')

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Linear Regression')
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

# Load and process the data
data_loader = DataLoader()
data = data_loader.load_data("../datasets/data_for_lr.csv")

# training dataset and labels
train_input = np.array(data['x'][0:500]).reshape(500, 1)
train_output = np.array(data['y'][0:500]).reshape(500, 1)

# valid dataset and labels
test_input = np.array(data['x'][500:700]).reshape(199, 1)
test_output = np.array(data['y'][500:700]).reshape(199, 1)

# Create and train the model

model = LinearRegression(train_input, train_output)
parameters, bias, loss = model.train(learning_rate=0.0001, iters=50)


# Evaluate the model
accuracy = model.accuracy(test_input, test_output)



print(f'Accuracy: {accuracy}')
print(f'Predicted value: {model.predict(24.0)}')

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import ctypes

# lib = ctypes.CDLL('./linear_regression.dll')
# lib.create_model.restype = ctypes.c_void_p
# lib.train_model.argtypes = [ctypes.c_int]

# lib.destroy_model.argtypes = [ctypes.c_void_p]

# lib.train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), 
#                        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float]

# lib.predict.restype = ctypes.c_float
# lib.predict.argtypes = [ctypes.c_void_p, ctypes.c_float]

# def numpy_to_float_array(data):
#     array_type = ctypes.c_float * data.size
#     return array_type(*data.flatten())

# class DataLoader():
#     def load_data(self):
#         # Load the dataset
#         url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
#         data = pd.read_csv(url)

#         # Drop the missing values
#         data = data.dropna()

#         return data

#     def split_data(self, data, train_ratio=0.8):
#         # Split the data into training and testing sets
#         train_size = int(len(data) * train_ratio)
#         train_data = data[:train_size]
#         test_data = data[train_size:]

#         return train_data, test_data

# class LinearRegression():
#     def __init__(self, train_inputs, train_outputs):
#         self.model = lib.create_model(train_inputs.shape[1])  

#         # Convert NumPy arrays to C arrays
#         self.train_inputs_c = numpy_to_float_array(train_inputs)
#         self.train_outputs_c = numpy_to_float_array(train_outputs)
        
#         self.bias = 0
#         self.loss = []

#     def cost_func(self, prediction, train_output):
#         # Calculate the cost
#         return np.mean((prediction - train_output) ** 2)

#     def forward_propagation(self):
#         # Forward propagation
#         return np.dot(self.train_inputs, self.weights) + self.bias

#     def update_weights(self, prediction, train_output, learning_rate):
#         # Update the weights
#         print("train_inputs.T shape: ", self.train_inputs.T.shape)
#         print("prediction shape: ", prediction.shape)
#         print("train_output shape: ", train_output.shape)
#         print("prediction-train_output shape: ", (prediction - train_output).shape)
#         print("train_inputs shape: ", self.train_inputs.shape[0])

        
#         prediction = prediction.reshape(-1, 1)
#         train_output = train_output.reshape(-1, 1)

#         # Calculate the gradients
#         gradient_weights = np.dot(self.train_inputs.T, (prediction - train_output)) / self.train_inputs.shape[0]
#         gradient_bias = np.mean(prediction - train_output)

#         # Update weights and bias
#         self.weights -= learning_rate * gradient_weights
#         self.bias -= learning_rate * gradient_bias

#     def train(self, learning_rate, iters):
#         fig, ax = plt.subplots()
#         ax.scatter(self.train_inputs, self.train_outputs, label='Training data')
#         line, = ax.plot(self.train_inputs, self.forward_propagation(), color='red', label='Fit')

#         def update(i):
#             # Train the C model (this is where the C code is used)
#             lib.train(self.model, ctypes.cast(self.train_inputs_c, ctypes.POINTER(ctypes.c_float)), 
#                        ctypes.cast(self.train_outputs_c, ctypes.POINTER(ctypes.c_float)),
#                        self.train_inputs.size, 1, learning_rate)

#             # Get updated weights and bias (you may need to implement these in C and expose them to Python)
#             # ... (Get weights and bias from C code) ...

#             # Update the weights and bias in Python
#             self.weights = # Updated weights from C
#             self.bias = # Updated bias from C

#             prediction = self.forward_propagation()
#             cost = self.cost_func(prediction, self.train_outputs)
#             print("w, b, c: ", self.weights, self.bias, cost)
#             self.loss.append(cost)
#             line.set_ydata(prediction)
#             return line,

#         ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
#         ani.save('linear_regression_A.gif', writer='ffmpeg')

#         plt.xlabel('Input')
#         plt.ylabel('Output')
#         plt.title('Linear Regression')
#         plt.legend()
#         plt.show()

#         return self.weights, self.bias, self.loss
        

#     def predict(self, test_input):
#         # Predict using C model
#         return lib.predict(self.model, test_input[0]) 

#     def accuracy(self, test_input, test_output):
#         # Calculate the accuracy
#         prediction = self.predict(test_input)
#         return 1 - self.cost_func(prediction, test_output) / np.mean(test_output ** 2)
    
#     def __del__(self):
#         # Free the C model
#         lib.destroy_model(self.model)

# # Load and process the data
# data_loader = DataLoader()
# data = data_loader.load_data()

# # training dataset and labels
# train_input = np.array(data['x'][0:500]).reshape(500, 1)
# train_output = np.array(data['y'][0:500]).reshape(500, 1)

# # valid dataset and labels
# test_input = np.array(data['x'][500:700]).reshape(199, 1)
# test_output = np.array(data['y'][500:700]).reshape(199, 1)

# # Create and train the model

# model = LinearRegression(train_input, train_output)
# parameters, bias, loss = model.train(learning_rate=0.0001, iters=50)


# # Evaluate the model
# accuracy = model.accuracy(test_input, test_output)



# print(f'Accuracy: {accuracy}')
# print(f'Predicted value: {model.predict(24.0)}')
