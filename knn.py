import csv
import random
import math
import operator
import pandas as pd

class KNN:

    def __init__(self, K):
        self.K = K
        
    # load a csv file
    def load_csv(self, filename):
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile, delimiter = ',')
            rows = list()
            for row in lines:
                rows.append(row[0])
            dataset = [rows[i].split(',') for i in range(len(rows))]
            return dataset
        
    # Converting features to float
    def str2float(self, dataset):
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = float(row[i])
        return dataset
            
    # Converting classes to int
    def str2int(self, dataset):
        classes = dict()
        class_str = [row[-1] for row in dataset]
        unique = set(class_str)
        for i, value in enumerate(unique):
            classes[value] = i
        classes_int = [classes[row[-1]] for row in dataset]
        return classes_int, class_str
    
    # Normalizing features
    def min_max(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax
    
    # Normalize the input of respective feature to the range 0-1
    def norm01(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
                
    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        accuracy = 100*correct/float(len(actual))
        return accuracy
    
    # Calculate the distance between two samples
    def find_dist(self, row, sample):
        distance = math.sqrt(sum([(row[i] - sample[i])**2 for i in range(len(row)-1)]))
        return distance
    
    # Find k nearest neighbors
    def find_knn(self, dataset, unknown_class, num_neighbors):
        distances = list()
        for row in dataset:
            dist = self.find_dist(row, unknown_class)
            distances.append((row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # Make a prediction with neighbors
    def predict_classification(self, dataset, unknown_class, num_neighbors):
        neighbors = self.find_knn(dataset, unknown_class, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction

    # Test kNN Algorithm
    def test_knn(self, dataset, test, num_neighbors):
        predictions = list()
        for row in test:
            output = self.predict_classification(dataset, row, num_neighbors)
            predictions.append(output)
        return(predictions)
    
    # k-fold cross-validation
    def kfold_cv(self, dataset, k_fold, num_neighbors):
        fold_size = int(len(dataset)/k_fold)
        test = list()
        train = list()
        dataset_split = list(dataset)
        actual = list()
        scores = list()
        i = 0
        while dataset_split:
            batch_size = fold_size
            if len(dataset_split) < 2*fold_size:
                batch_size = len(dataset_split)
            test.append(random.sample(dataset_split, batch_size))
            temp_dataset = list(dataset)
            temp_actual = list()
            for j in range(len(test[i])):
                temp_dataset.remove(test[i][j])
                dataset_split.remove(test[i][j])
                temp_actual.append(test[i][j][-1])
            train.append(temp_dataset)
            actual.append(temp_actual)
            predicted = self.test_knn(train[i], test[i], num_neighbors)
            scores.append(self.accuracy_metric(actual[i], predicted))
            i += 1
        return scores
            
