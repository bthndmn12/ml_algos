import zlib
import numpy as np
import pandas as pd
import datasets

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
    

class NCD_Classifier:
    def __init__(self, k=3):
        self.k = k

    def normalized_compression_distance(self, x1, x2):
        """Calculates the Normalized Compression Distance between two strings.

        Args:
          x1: The first string.
          x2: The second string.

        Returns:
          The NCD value between x1 and x2.
        """
        Cx1 = len(zlib.compress(x1.encode()))
        Cx2 = len(zlib.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(zlib.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        return ncd

    def predict_class(self, text, dataset):
        """Predicts the class of a given text using the NCD algorithm and KNN.

        Args:
          text: The text to classify.
          dataset: A list of dictionaries, where each dictionary has keys 'text' and 'label'.

        Returns:
          The predicted class label.
        """
        distances = []
        for data_point in dataset:
            question = data_point['text']
            answer = data_point['label']
            distance = self.normalized_compression_distance(text, question)
            distances.append((distance, answer))
        distances.sort(key=lambda x: x[0])
        top_k_class = [distances[i][1] for i in range(self.k)]
        predicted_class = max(set(top_k_class), key=top_k_class.count)
        return predicted_class


dataset = datasets.load_dataset("winvoker/turkish-sentiment-analysis-dataset", split="train")
questions = dataset['text']
answers = dataset['label']

data_tuples = list(zip(questions, answers))

train_size = int(0.8 * len(data_tuples))
training_set = data_tuples[:train_size]
test_set = data_tuples[train_size:]

sample_text = "Çok iyi bir cihaz, tavsiye ediyorum."
knn = NCD_Classifier(k=3)
predicted_answer = knn.predict_class(sample_text, dataset)
print("Sample Text:",sample_text)
print("Predicted Answer:",predicted_answer)


# data_loader = DataLoader()
# data = data_loader.load_data("../../datasets/Social_Network_Ads.csv")
# # print(data.data.head())
# print(data.head())


# data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})


# train_input = np.array(data[['Gender','Age','EstimatedSalary']][0:300]).reshape(300, 3)
# train_input_strings = [",".join(str(x) for x in row) for row in train_input]

# test_input = np.array(data[['Gender','Age','EstimatedSalary']][300:400]).reshape(100, 3)
# test_input_strings = [",".join(str(x) for x in row) for row in test_input] 

# train_output = np.array(data['Purchased'][0:300]).reshape(300, 1)
# test_output = np.array(data['Purchased'][300:400]).reshape(100, 1)

# train_set = [(train_input_strings[i], str(train_output[i][0])) for i in range(len(train_input))]
# test_set = [(test_input_strings[i], str(test_output[i][0])) for i in range(len(test_input))]

# knn = NCD_Classifier(k=3)
# predicted_classes = knn.predict_class(test_set, train_set)
# print(predicted_classes)

# Calculate Accuracy
# correct_predictions = 0
# for i in range(len(test_set)):
#   if predicted_classes[i] == test_set[i][1]:
#     correct_predictions += 1

# accuracy = correct_predictions / len(test_set)
# print(f"Accuracy: {accuracy}")

#------------------------------------------------------------

# class NCD_Classifier:
#     def __init__(self, k=3):
#         self.k = k

#     def normalized_compression_distance(self, x1, x2):
#         """Calculates the Normalized Compression Distance between two data points.

#         Args:
#           x1: The first data point (can be a string, list, or any object that can be encoded).
#           x2: The second data point (should be of the same type as x1).

#         Returns:
#           The NCD value between x1 and x2.
#         """
#         Cx1 = len(zlib.compress(str(x1).encode()))  # Encode data points to strings
#         Cx2 = len(zlib.compress(str(x2).encode()))
#         x1x2 = " ".join([str(x1), str(x2)])
#         Cx1x2 = len(zlib.compress(x1x2.encode()))
#         ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
#         return ncd

#     def predict_class(self, data_point, dataset):
#         """Predicts the class of a given data point using the NCD algorithm and KNN.

#         Args:
#           data_point: The data point to classify.
#           dataset: A list of tuples, where each tuple is (data_point, label).

#         Returns:
#           The predicted class label.
#         """
#         distances = []
#         for item in dataset:
#             example, label = item
#             distance = self.normalized_compression_distance(data_point, example)
#             distances.append((distance, label))
#         distances.sort(key=lambda x: x[0])
#         top_k_class = [distances[i][1] for i in range(self.k)]
#         predicted_class = max(set(top_k_class), key=top_k_class.count)
#         return predicted_class
    






