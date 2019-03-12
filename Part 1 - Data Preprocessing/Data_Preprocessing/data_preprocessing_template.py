# Data Preprocessing

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')
# Alocate all the lines, and all the columns except the last one to a matrix
x = dataset.iloc[:, :-1].values
# Alocate all the lines, and only the last column to a matrix
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# Define strategy
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# Read X and then use the strategy to update X
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])



# Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# For X:
labelencoder_x = LabelEncoder()
# Update the collumn 0 of X with the encoding
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

# For Y:
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)