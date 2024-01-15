# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 00:32:08 2024

@author: HP
"""
#------------Instructions---------------
#1 Import you data and perform basic data exploration phase
#i Display general information about the dataset
#ii Create a pandas profiling reports to gain insights into the dataset
#iii Handle Missing and corrupted values
#iv Remove duplicates, if they exist
#v Handle outliers, if they exist
#vi Encode categorical features
#2 Select your target variable and the features
#3 Split your dataset to training and test sets
#4 Based on your data exploration phase select a ML regression algorithm and train it on the training set
#5 Assess your model performance on the test set using relevant evaluation metrics
#6 Discuss with your cohort alternative ways to improve your model performance
# -----------SOLUTION--------
#1 Import you data and perform basic data exploration phase

# ---------SOLUTION

#1 -----------Import you data and perform basic data exploration phase
import pandas as pd
import sweetviz as sv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


data = "C:/Users/HP/Downloads/5G_energy_consumption_dataset_visualization.csv"
dataset = pd.read_csv(data)

# Converting date column to the date and time format
dataset["Time"] = pd.to_datetime(dataset["Time"], errors='coerce')

# Handling missing or incorrect values
dataset["Time"].fillna(pd.to_datetime('today'), inplace=True)

# Removing the time part from a Date coulmn in Pandas
dataset["Time"] = dataset["Time"].dt.date

#i Display general information about the dataset
dataset.info()
describe_data = dataset.describe()

# #ii Create a pandas profiling reports to gain insights into the dataset
profile_report = sv.analyze(dataset)
profile_report.show_html('profile_report.html')

#iii Handle Missing and corrupted values
missing = dataset.isnull().sum().sum() # counting the NAN

#iv Remove duplicates, if they exist
# Identifying Duplicate Records in a Pandas DataFrame and Counting
duplicate = (dataset.duplicated().sum())
# Removing Duplicate Data in a Pandas DataFrame
dataset.drop_duplicates(inplace = True)

#v Handle outliers, if they exist
# Identifing the Outliers Using Visual Analysis
sns.boxplot(x=dataset['Energy'])
plt.show()

#vi Encode categorical features
# Using LabelEncoder
label_encoder = LabelEncoder()
dataset['BS'] = label_encoder.fit_transform(dataset['BS'])


#2 -----------Select your target variable and the features
target_variable = dataset["BS"]
f_column = ["Energy", "load", "TXpower"]
features = dataset[f_column]

#3------------Split your dataset to training and test sets
ds = dataset
X = dataset.drop(["BS", "Time"], axis=1)  # Features
y = dataset["BS"]  # Target variable

# Splitting the data into training and testing sets (default is 75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


#4 -----------Based on your data exploration phase select a ML regression algorithm and train it on the training set
model = LinearRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)


#5------------Assess your model performance on the test set using relevant evaluation metrics
# Evaluating Model Perfomance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# A Visual Report of The Model's Prediction

# Assuming 'y_test' is the actual target values and 'y_pred' is the predicted values
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()








