
"""
Example RF Analysis
This script builds a very simple RandomForestRegressor from the sci-kitlearn 
library and shows how to do some basic analysis on it.
"""
#==============================================================================
#Imports, globals, filepaths, loading data

import pandas as pd
import numpy as np
import xarray as xr

startyear = 1979
endyear = 1995

folder = '~/Desktop/IMAS/'  #filepath
pfile = 'Proxy_combined_data_v4.nc'  #proxy filename
df = xr.open_dataset(folder+pfile).sel(year= slice(startyear, endyear))
df = df.mean('lon').mean('month')
df = df.to_dataframe()
df.reset_index(inplace=True)


#==============================================================================
#Preparing data

mean_sie = df['SIE'].mean()

X = df.iloc[:, 2:8].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

#Standardise values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#==============================================================================
#Training model

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


#==============================================================================
#Evaluate model

from sklearn import metrics

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse)
print('RMSE % of mean:', 100*rmse/mean_sie)

#==============================================================================
#Analysing RF patterns

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig = plt.figure(figsize = (15,10))
plot_tree(regressor.estimators_[0],
          feature_names = df.iloc[:, 2:8].columns,
          class_names = df['SIE'].unique(),
          filled = True, rounded = True)

plt.show()


#==============================================================================
#Feature importances
importances = pd.Series(regressor.feature_importances_, index=df.iloc[:, 2:8].columns)
importances = importances.sort_values(ascending=False)
print(importances)