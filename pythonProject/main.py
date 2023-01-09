# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random
from _csv import reader
from math import sqrt
from pathlib import Path

import pandas

def modifyDataset(dataset):
    dataset.iloc[1:]
    print(dataset)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 16].values

    from sklearn import preprocessing
    print(dataset.columns)
    le = preprocessing.LabelEncoder()
    for column_name in dataset.columns:
        if column_name == dataset.columns[16]:
            break
        if dataset[column_name].dtype == object:
            dataset[column_name] = le.fit_transform(dataset[column_name])
        else:
            pass

    filepath = Path("C:\\Users\\patra\\PycharmProjects\\pythonProject\\modifiedData.csv")
    dataset.to_csv(filepath, index=False)

def sklearnSolution(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 16].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=8)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import classification_report, accuracy_score

    result1 = classification_report(y_test, y_pred)
    result2 = accuracy_score(y_test, y_pred)
    return result1, result2

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def euclidean_distance(row1, row2):
 distance = 0.0
 for i in range(len(row1)-1):
    distance += (row1[i] - row2[i])**2
 return sqrt(distance)

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
		print('[%s] => %d' % (value, i))
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

headernames = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import pandas as pd

    resultsMap = {
        0 : 'Overweight_Level_II',
        1 : "Normal_Weight",
        2 : "Insufficient_Weight",
        3 : "Overweight_Level_I",
        4 : "Obesity_Type_III",
        5 : "Obesity_Type_I",
        6 : "Obesity_Type_II",
    }

    dataset = pd.read_csv("C:\\Users\\patra\\PycharmProjects\\pythonProject\\modifiedData.csv", names=headernames)
    # modifyDataset(dataset)
    # dataset = modifyDataset(dataset) this was used to encode the string values using LabelEncoder
    dataset = dataset.iloc[1:]

    skResult1, skResult2 = sklearnSolution(dataset)
    print("sklearn Classification Report:", )
    print(skResult1)
    print("sklearn Accuracy:", skResult2 * 100, "%")

    dataset = load_csv("C:\\Users\\patra\\PycharmProjects\\pythonProject\\modifiedData.csv")
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    #number of neighbours
    k_num = 8
    testNr = len(dataset)//40
    trainNr = len(dataset) - testNr
    train_data = random.sample(dataset, trainNr)
    test_data = random.sample(dataset, testNr)


    testResults = []
    testExpectedResults = []
    for x in test_data:
        testExpectedResults.append(x[-1])
        label = predict_classification(dataset, x, k_num)
        testResults.append(label)

    from sklearn.metrics import classification_report, accuracy_score

    for i in range(0, len(testResults)):
        testResults[i] = resultsMap[ testResults[i] ]
        testExpectedResults[i] = resultsMap[ testExpectedResults[i] ]

    result1 = classification_report(testExpectedResults, testResults)
    result2 = accuracy_score(testExpectedResults, testResults)

    print("Manual Classification Report:", )
    print(result1)
    print("Manual Accuracy:", result2 * 100, "%")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
