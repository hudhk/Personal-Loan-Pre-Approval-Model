import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

# Importing data
data = pd.read_csv('A2-UniBank.csv')

# Clearing and Preprocessing Data for Naive Bayes & Decision Tree 

print(data.isnull().sum())
data_cleaned = data.dropna()
print(data_cleaned.shape)
data = data.iloc[:, :14]
print(data.head())

# Defining Data Types and Continuous Variables Required for Analysis

data_types = data.dtypes
continuous_variables = data_types[data_types != 'object'].index.tolist()

print(continuous_variables)

# Once Continuous Variables Defined --> Gathering Avg, Mean, Min, and Max values
# Utilizing Defined Variables Determined Num of Bins Required
cont_variables = ['Income', 'CCAvg', 'Mortgage']
data_cont = data[cont_variables]

for variable in cont_variables:
    print("Variable:", variable)
    print("Average:", data[variable].mean())
    print("Mean:", data[variable].median())
    print("Minimum:", data[variable].min())
    print("Maximum:", data[variable].max())
    print()

num_Income_bins = 5
num_CCAvg_bins = 3
num_Mortgage_bins = 4

# Creating Variable Bin Column to begin train_test_split processing using KBinsDiscretizer

Income_binner = KBinsDiscretizer(n_bins=num_Income_bins, encode='ordinal', strategy='quantile')
data['Income_bins'] = Income_binner.fit_transform(data_cont[['Income']])

CCAvg_binner = KBinsDiscretizer(n_bins=num_CCAvg_bins, encode='ordinal', strategy='quantile')
data['CCAvg_bins'] = CCAvg_binner.fit_transform(data_cont[['CCAvg']])

Mortgage_binner = KBinsDiscretizer(n_bins=num_Mortgage_bins, encode='ordinal', strategy='quantile')
data['Mortgage_bins'] = Mortgage_binner.fit_transform(data_cont[['Mortgage']])

print(data.head())

# Setting train_test_splits 

validation_ratio = 0.3
test_ratio = 0.2

validation_size = int(len(data) * validation_ratio)
test_size = int(len(data) * test_ratio)

train_data, temp_data = train_test_split(data, test_size=validation_size + test_size, random_state =100)
validation_data, test_data = train_test_split(temp_data, test_size=test_size, random_state=100)


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#Classification utilizing Naive Bayes

x = data[['Income_bins', 'CCAvg_bins', 'Mortgage_bins', 'Education']]
y = data['Personal Loan']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

y_pred = nb_classifier.predict(x_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

y_prob = nb_classifier.predict_proba(x_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

#Classification Using Decision Tree

from sklearn.tree import DecisionTreeClassifier

depths = []
accuracies = []
max_depths = range(1, 21)

for max_depth in max_depths:
    dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt_classifier.fit(x_train, y_train)

    y_pred = dt_classifier.predict(x_test)

    accuracy = np.mean(y_pred == y_test)

    depths.append(max_depth)
    accuracies.append(accuracy)

print("Partition of y Data:")
print("Train Set (y_train):", y_train.shape)
print("Train Set (y_test):", y_test.shape)

print("\nHistory of Train/Test Split:")
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print("\nFit Details:")
print("Max Depth of Decision Tree:", max_depth)
print("Number of Features:", x_train.shape[1])

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color ='darkorange', lw=2, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Naive Bayes')
plt.legend(loc='lower right')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(depths, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Maximum Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.title('Decision Tree Split History')
plt.grid(True)
plt.show()