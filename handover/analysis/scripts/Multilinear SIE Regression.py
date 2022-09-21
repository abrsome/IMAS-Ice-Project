
#SIE Multilinear Regression by season and longitude
#==============================================================================
#Imports, globals, filepaths, loading data

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn import metrics

startyear = 1979
endyear = 1995
yvar = 'SIE'

folder = '~/Desktop/IMAS/'  #filepath
pfile = 'Proxy_combined_data_v4.nc'  #proxy filename
df = xr.open_dataset(folder+pfile).sel(year= slice(startyear, endyear))

#==============================================================================
#Preparing data

#Array to store root mean square error as percentage of mean
rmsepom = np.zeros([12,36]) #Initialise an array of correlations

#1**: Do mean SIE globally
mean_sie = df['SIE'].mean('month').mean('lon').mean('year')

years = np.arange(startyear, endyear+1)
testyears = np.sort(np.random.choice(years, size = int(len(years)*0.33), replace = False))
trainyears = [x for x in years if x not in testyears]

#Proxies data TS
X = df[(list(df.keys())[1:])].mean('month')
X_train = X.sel(year = trainyears).to_dataframe().values
X_test = X.sel(year = testyears).to_dataframe().values

#Standardise values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Loop over season/longitude
for m in range(12): #loop over the months
    m+=1
    months = [(m-3)%12+1, (m-2)%12+1, (m-1)%12+1] #nonneg month ints for season ending in m
    monat = df[yvar].sel(month=months) #overlapping years w/ WHG    
    monat = monat.mean('month')
    for l in np.arange(0.25,360,10): #loop over the longitudes
        y = monat.sel(lon= slice(l, l+10)).mean(dim='lon')
        y_train = y.sel(year = trainyears).to_dataframe().values
        y_test = y.sel(year = testyears).to_dataframe().values
        
        #Training model
        from sklearn.linear_model import LinearRegression
        mlr = LinearRegression()
        mlr.fit(X_train, y_train)
        y_pred = mlr.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        
        #1**: Do mean SIE local to test set
        #mean_sie = y_test.mean()
        
        rmsepom[m-1,int((l-0.25)/10)] = 100*rmse/mean_sie

#==============================================================================
#Analysis

#print("=========================================================")
print('Error in '+yvar+' predictions using Multi Linear Regression')
print('RMSE % minimum: ', rmsepom.min())
print('RMSE % maximum: ', rmsepom.max())
print('RMSE % mean: ', rmsepom.mean())


#Plot all the way to 30:
ax = plt.subplot()

clev = np.arange(0,30,2)  #contour levels
rmsepomxr = xr.DataArray(rmsepom)
rmsepomxr.plot.pcolormesh(levels = clev,cmap = 'cividis')

ax.set_title('Error in '+yvar+' predictions using Multi Linear Regression')
ax.set_ylabel('Season')
ax.set_xlabel('Longitude')
seasons = ['OND','NDJ','DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']
ax.set_xticks(ticks=np.arange(3,36,3), labels=['','60E','','120E','','180','','120W','','60W',''])
ax.set_yticks(ticks=np.arange(-0.5,12,1), labels=seasons, rotation=0)

plt.show()


#Plot to 5:
ax = plt.subplot()

clev = np.arange(0,5,.25)  #contour levels
rmsepomxr = xr.DataArray(rmsepom)
rmsepomxr.plot.pcolormesh(levels = clev, cmap = 'cividis')

ax.set_title('Error in '+yvar+' predictions using Multi Linear Regression')
ax.set_ylabel('Season')
ax.set_xlabel('Longitude')
seasons = ['OND','NDJ','DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']
ax.set_xticks(ticks=np.arange(3,36,3), labels=['','60E','','120E','','180','','120W','','60W',''])
ax.set_yticks(ticks=np.arange(-0.5,12,1), labels=seasons, rotation=0)

plt.show()

