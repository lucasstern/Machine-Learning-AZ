# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(X_test)

# Building the optimal model using backward elimination
import statsmodels.formula.api as sm
# Making a column with just ones, so the statsmodel librarie undestand the b0,
# on the multiple linear regression formula
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Step 1 - Define Significance Level
SL = 0.05
# Step 2 - Fit
regressorOLS = sm.OLS(endog=y, exog=X_opt).fit()
# Step 3 - Get the predictor with the highest P-Value
regressorOLS.summary()
# Step 4 - Remove the IV found on step 3 and repeat step 2 until
# theres no more P < SL
X_opt = X[:, [0, 1, 3, 4, 5]]
regressorOLS = sm.OLS(endog=y, exog=X_opt).fit()
regressorOLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressorOLS = sm.OLS(endog=y, exog=X_opt).fit()
regressorOLS.summary()
X_opt = X[:, [0, 3, 5]]
regressorOLS = sm.OLS(endog=y, exog=X_opt).fit()
regressorOLS.summary()
X_opt = X[:, [0, 3]]
regressorOLS = sm.OLS(endog=y, exog=X_opt).fit()
regressorOLS.summary()


# Automatic version
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



 