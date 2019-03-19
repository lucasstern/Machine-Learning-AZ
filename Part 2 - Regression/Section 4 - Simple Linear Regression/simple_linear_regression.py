# Simples Linear Regression

# Importing Essential Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values # Independent Variable - Years
y = dataset.iloc[:, 1].values  # Dependent Variable(target variables) - Salaries

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3,
                                                    random_state=0)

# Feature Scalling -> The LinearRegrssion below will do that for us
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red') # Real values
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Prediction Line
plt.title('Salário vs Experiência (Training Set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()

# Visualising the Testing set results
plt.scatter(X_test, y_test, color = 'red') # Real values
# Prediction Line - Dá na mesma :)
plt.plot(X_train, regressor.predict(X_train), color = 'blue') 
plt.title('Salário vs Experiência (Test Set)')
plt.xlabel('Anos de Experiência')
plt.ylabel('Salário')
plt.show()
