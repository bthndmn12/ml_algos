import gzip
import numpy as np
import pandas as pd


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

class KNNClassifier():
    def __init__(self, k):
        self.k = k

    def predict_class(self, test_set, training_set):
        predicted_classes = []

        for x1, _ in test_set:
            Cx1 = len(gzip.compress(x1.encode()))
            distance_from_x1 = []

            for x2, _ in training_set:
                Cx2 = len(gzip.compress(x2.encode()))
                x1x2 = " ".join([x1, x2])
                Cx1x2 = len(gzip.compress(x1x2.encode()))
                ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
                distance_from_x1.append(ncd)

            sorted_idx = np.argsort(np.array(distance_from_x1))[:self.k]
            top_k_class = [training_set[i][1] for i in sorted_idx]
            predict_class = max(set(top_k_class), key=top_k_class.count)
            predicted_classes.append(predict_class)

        return predicted_classes


data_loader = DataLoader()
data = data_loader.load_data("../../datasets/Social_Network_Ads.csv")
# print(data.data.head())
print(data.head())


data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})


train_input = np.array(data[['Gender','Age','EstimatedSalary']][0:300]).reshape(300, 3)
train_input_strings = [",".join(str(x) for x in row) for row in train_input]

test_input = np.array(data[['Gender','Age','EstimatedSalary']][300:400]).reshape(100, 3)
test_input_strings = [",".join(str(x) for x in row) for row in test_input] 

train_output = np.array(data['Purchased'][0:300]).reshape(300, 1)
test_output = np.array(data['Purchased'][300:400]).reshape(100, 1)

train_set = [(train_input_strings[i], str(train_output[i][0])) for i in range(len(train_input))]
test_set = [(test_input_strings[i], str(test_output[i][0])) for i in range(len(test_input))]

knn = KNNClassifier(k=3)
predicted_classes = knn.predict_class(test_set, train_set)
print(predicted_classes)

# accuracy
correct_predictions = 0
for i in range(len(test_set)):
  if predicted_classes[i] == test_set[i][1]:
    correct_predictions += 1

accuracy = correct_predictions / len(test_set)
print(f"Accuracy: {accuracy}") # gives 60 percent accuracy





