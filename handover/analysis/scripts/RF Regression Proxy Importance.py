
#Showing how to do proxy importance for RF regression
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

folder = '~/Desktop/IMAS/ncfiles/'  #filepath
pfile = 'Proxy_combined_data_v4.nc'  #proxy filename
df = xr.open_dataset(folder+pfile).sel(year= slice(startyear, endyear))
columns = (list(df.keys())[1:])


#==============================================================================
#Preparing data

years = np.arange(startyear, endyear+1)
testyears = np.sort(np.random.choice(years, size = int(len(years)*0.33), replace = False))
trainyears = [x for x in years if x not in testyears]

#Array to store importance arrays
importancesarray = [[] for _ in range(len(columns))]
importancesarray[0] = np.zeros([12,36])
for p in range(len(columns)):
    importancesarray[p] = np.zeros([12,36])

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
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
        regressor.fit(X_train, y_train)
        importances = pd.Series(regressor.feature_importances_, index=columns)
        for p in range(len(columns)):
            importancesarray[p][m-1,int((l-0.25)/10)] = importances[p]
            

#==============================================================================
#Heatmaps
maximportances = np.zeros(len(columns))
for i in range(len(columns)):
    maximportances[i] = importancesarray[i].max()

#Plot the heatmaps:
for p in range(len(columns)):
    ax = plt.subplot()
    
    clev = np.arange(0,round(maximportances.max(),1)+.1,.1)  #contour levels
    importancexr = xr.DataArray(importancesarray[p])
    importancexr.plot.pcolormesh(levels = clev, cmap = 'inferno')
    
    ax.set_title(columns[p] +' importance in in SIE RF prediction.')
    ax.set_ylabel('Season')
    ax.set_xlabel('Longitude')
    seasons = ['OND','NDJ','DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']
    ax.set_xticks(ticks=np.arange(3,36,3), labels=['','60E','','120E','','180','','120W','','60W',''])
    ax.set_yticks(ticks=np.arange(-0.5,12,1), labels=seasons, rotation=0)
    
    plt.show()
