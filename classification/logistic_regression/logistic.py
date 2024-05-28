import numpy as np
import pandas as pd
import sys
sys.path.append('../datasets')
from synthetic_data import SyntheticData
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt



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
class LogisticRegression():

    def __init__(self, train_inputs, train_outputs, learning_rate=0.001, epochs=1000) -> None:
        self.train_inputs = self.normalize(train_inputs)
        self.train_outputs = train_outputs.reshape(-1, 1)
        self.weights = np.random.randn(train_inputs.shape[1], 1)
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def normalize(self, inputs):
        return (inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0)

    # def normalize(self, X):
    # # If X is 1-dimensional, reshape it to 2-dimensional
    #     if len(X.shape) == 1:
    #         X = X.reshape(-1, 1)

    #     X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)

    #     return X

    def sigmoid(self, z):
        # z = np.clip(z, -500, 500) 
        return 1 / (1 + np.exp(-z))
    
    def logit(self, inputs):
        return np.dot(inputs, self.weights) + self.bias
    
    def forward_propagation(self, train_inputs):
        return self.sigmoid(self.logit(train_inputs))
    
    def cost_func(self, prediction, train_output):
        epsilon = 1e-10  
        prediction = np.clip(prediction, epsilon, 1 - epsilon)
        return -np.mean(train_output * np.log(prediction) + (1 - train_output) * np.log(1 - prediction))
    
    def update_weights(self):
        predictions = self.forward_propagation(self.train_inputs)
        m = self.train_inputs.shape[0]
        dw = np.dot(self.train_inputs.T, (predictions - self.train_outputs)) / m
        db = np.sum(predictions - self.train_outputs) / m
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def train(self):
        for epoch in range(self.epochs):
            self.update_weights()
            if epoch % 100 == 0: 
                predictions = self.forward_propagation(self.train_inputs)
                cost = self.cost_func(predictions, self.train_outputs)
                print(f"Epoch {epoch}, Cost: {cost}")

    def predict(self, test_inputs):
        normalized_test_inputs = self.normalize(test_inputs)
        pred = self.forward_propagation(normalized_test_inputs)
        for i in range(len(pred)):
            if pred[i] > 0.5:
                print("pred and test outputs", 1, self.train_outputs[i])
            else:
                print("pred and test outputs", 0, self.train_outputs[i])
            
        return pred
    
    def accuracy(self, test_inputs, test_outputs):
        predictions = self.predict(test_inputs)
        return np.mean((predictions > 0.5) == test_outputs)



# train_inputs = np.array(...)  # Your training inputs
# train_outputs = np.array(...)  # Your training outputs
# data = SyntheticData(SyntheticData.DataType.LOGARITHMIC, 2000, 0.1,10)
# sin_data = data.data
data_loader = DataLoader()
data = data_loader.load_data("../datasets/Social_Network_Ads.csv")
# print(data.data.head())
print(data.head())

train_input = np.array(data[['Age','EstimatedSalary']][0:300]).reshape(300, 2)
train_output = np.array(data['Purchased'][0:300]).reshape(300, 1)

test_input = np.array(data[['Age','EstimatedSalary']][300:400]).reshape(100, 2)
test_output = np.array(data['Purchased'][300:400]).reshape(100, 1)

model = LogisticRegression(train_input, train_output)
model.train()
print(test_input[0])
print(model.predict(test_input[0]))
print(test_output[0])

test_predictions = model.predict(test_input)
accuracy = model.accuracy(test_input, test_output)
print(f"Accuracy: {accuracy}")


plot_data = pd.DataFrame({
    'Age': test_input[:, 0],
    'EstimatedSalary': test_input[:, 1],
    'Purchased': test_output.flatten(),
    'Probability': test_predictions.flatten()  
})

plt.figure(figsize=(10, 6))  
sns.scatterplot(x='Age', y='EstimatedSalary', hue='Purchased', data=plot_data, s=50, palette='viridis')


sns.scatterplot(x='Age', y='EstimatedSalary', hue='Probability', data=plot_data, marker='x', s=100, palette='magma', legend='brief')

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Actual vs. Predicted Purchases (with Probability)")
plt.legend(title="Legend")
plt.show()

# import numpy as np

# class LogisticRegression():

#     def __init__(self, train_inputs, train_outputs) -> None:
#         self.train_inputs = train_inputs
#         self.train_outputs = train_outputs.reshape(-1, 1)
#         self.weights = np.zeros((train_inputs.shape[1], 1))
#         self.bias = 0.0

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))
    
#     def logit(self, inputs):
#         return np.dot(inputs, self.weights) + self.bias
    
#     def log_likelihood(self, prediction, train_output):
#         return np.sum(train_output * np.log(prediction) + (1 - train_output) * np.log(1 - prediction))
    

#     def forward_propagation(self):
#         return self.sigmoid(self.logit(self.train_inputs))
    
#     def cost_func(self, prediction, train_output):
#         return -np.mean(train_output * np.log(prediction) + (1 - train_output) * np.log(1 - prediction))
    
    