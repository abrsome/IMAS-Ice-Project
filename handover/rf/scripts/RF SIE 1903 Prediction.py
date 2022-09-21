
"""
Doing a monthly SIE prediction back to 1903 using month mode.
"""

#==============================================================================
#Imports, globals, filepaths, loading data

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from calendar import month_abbr 

months = list(month_abbr)
months.pop(0)

n_est = 200 #number of estimators
rs = 0 #random state
np.random.seed(rs)

predstartyear = 1903 #start of predicted period
startyear = 1979 #start of training
endyear = 2006 #end of training
predyears = startyear-predstartyear

yvar = 'SIE'
folder = '~/Desktop/IMAS/ncfiles/'  #filepath
pfile = 'Proxy_combined_data_v4.nc'  #proxy filename
xfile = 'CombinedProxies.nc'  #proxy filename
df = xr.open_dataset(folder+pfile).sel(year= slice(startyear, endyear))
X = xr.open_dataset(folder+xfile).sel(year= slice(predstartyear, endyear))

#------------------------------------------------------------------------------
"""
Some datasets have missing values. We have two options:
    1. Remove the predictor entirely if it contains a nan in the time range.
    2. Use a scaler to interpolate nan values. (not yet implemented)
"""
#1. To drop predictors containing NaN in this range:
#Doesn't look like xarray has equivalent of dropna(axis=1) so I am going
#through pandas and back to xarray:
X = X.to_dataframe().dropna(axis=1).to_xarray()

#2. To be implemented.
"""
Using month mode of prediction:
Month = consider data from all longitudes in a month as one target
"""
outputname = ['Individual','Month','Longitudinal','Year']
mode = 1

#==============================================================================
#Preparing data

#Global SIE mean
mean_sie = df[yvar].mean('month').mean('lon').mean('year')
"""
Option here to use different means, respective to the year, month, long, etc.
"""

#------------------------------------------------------------------------------
"""
#Generating a train test split. Doing this here with randomly selecting arrays 
# of years instead of sklearn traintest split as not all the splits are done
# simultaneously and therefore can't all be done in one call of the function.
# So, need to generate year lists here to ensure years in data are consistent
# across X and y's.
"""
years = np.arange(startyear, endyear+1)
testyears = np.sort(np.random.choice(years, size = int(len(years)*0.33), replace = False))
trainyears = [x for x in years if x not in testyears]

#------------------------------------------------------------------------------
#Proxies data into time series
X_train = X.sel(year = trainyears).to_dataframe().values
X_test = X.sel(year = testyears).to_dataframe().values

#X_reg is the predictors in the regression time period
X_reg = X.sel(year = slice(predstartyear,startyear-1)).to_dataframe().values

#Standardise values of proxies
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_reg = sc.transform(X_reg)


#------------------------------------------------------------------------------
#Get SIE into bins of 10degree longitude sections
y = df[yvar].groupby_bins('lon', np.arange(0,361,10)).mean()

#==============================================================================
#Training and testing model

#Array to store root mean square error as percentage of mean
#RMSEPOM = Root Mean Squared Error as Percentage Of Mean
rmsepom = np.zeros([12,36])
regression = np.zeros([predyears,12,36])

    
for m in range(12): #loop over the months
    m += 1 #month index in xarray starts at 1
    monthdata = y.sel(month=m)
    y_train = monthdata.sel(year = trainyears).to_numpy()
    y_test = monthdata.sel(year = testyears).to_numpy()
    
    #Training model
    regressor = RandomForestRegressor(n_estimators = n_est, random_state = rs)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    y_reg = regressor.predict(X_reg) #yr=0 for 1903
    
    #Convert nested arrays from longitudes to year time series
    y_test = y_test.transpose()
    y_pred = y_pred.transpose() 
    
    for long in range(36):
        rmse = np.sqrt(metrics.mean_squared_error(y_test[long], y_pred[long]))
        rmsepom[m-1,long] = 100*rmse/mean_sie
    
    for yr in range(predyears):
        regression[yr,m-1] = y_reg[yr]

        
        
#==============================================================================
#draw function for analysis
def draw(rmsepom):
    ax = plt.subplot()
    
    rmsepomxr = xr.DataArray(rmsepom)
    rmsepomxr.plot.pcolormesh(levels = clev)
    
    ax.set_title('Error in '+yvar+' predictions using RFs')
    ax.set_ylabel('Month')
    #plt.yticks(ticks=np.arange(12))
    ax.set_xlabel('Longitude')
    ax.set_xticks(ticks=np.arange(3,36,3), labels=['','60E','','120E','','180','','120W','','60W',''])
    ax.set_yticks(ticks=np.arange(12), labels=months, rotation=0)
    ax.annotate("Mode = {}".format(outputname[mode]), xy = (0,0), xytext = (-6,-2.1))
    
    plt.show()
    
    
#==============================================================================
#draw function for the error distribution
def drawerror(rmsepom):
    ax = plt.subplot()
    
    ordered = rmsepom.copy()
    ordered.ravel().sort()
    
    x1 = np.percentile(ordered.ravel(), 50)
    plt.plot([x1, x1], [0, 100], color='g', linestyle='dashed', linewidth=2)
    ax.annotate("50%: {}".format(round(x1,1)), xy = (35,100), color = 'g')
    
    x1 = np.percentile(ordered.ravel(), 75)
    ax.annotate("75%: {}".format(round(x1,1)), xy = (35,92.5), color = 'tab:orange')
    plt.plot([x1, x1], [0, 100], color='tab:orange', linestyle='dashed', linewidth=2)
    
    x1 = np.percentile(ordered.ravel(), 95)
    plt.plot([x1, x1], [0, 100], color='r', linestyle='dashed', linewidth=2)
    ax.annotate("95%: {}".format(round(x1,1)), xy = (35,85), color='r')
    
    plt.hist(ordered.ravel(), bins = np.arange(0,41,2))
    
    ax.set_title('Error distribution for mode: {}'.format(outputname[mode]))
    ax.set_ylabel('Number of Values')
    ax.set_xlabel('Error%')
    ax.set_yticks(ticks=np.arange(0,120,10))
    ax.set_xticks(ticks=np.arange(0,41,2))
    ax.annotate("Estimators = {}".format(n_est), xy = (0,0), xytext = (-6,-20))
    
    plt.show()
    

#==============================================================================
#Analysis of model


print("=========================================================")
print('Error in '+yvar+' predictions using RFs with Output Mode = {}'.format(outputname[mode]))
print('RMSE % minimum: ', rmsepom.min())
print('RMSE % maximum: ', rmsepom.max())
print('RMSE % mean: ', rmsepom.mean())
print("=========================================================")



#Contour levels for 'zoomed out'
clev = np.arange(0,30,2)  #contour levels
draw(rmsepom)

#Contour levels for 5%
clev = np.arange(0,5,.25)  #contour levels
draw(rmsepom)

drawerror(rmsepom)


#==============================================================================
#Producing NetCDF format
prediction = xr.DataArray(regression,attrs=dict(description=("Prediction of SIE 1903 to 1978")))
prediction = prediction.rename({'dim_0': 'year', 'dim_1':'month','dim_2':'lon'})
prediction = prediction.assign_coords(year = np.arange(1903,1979))
prediction = prediction.assign_coords(month = np.arange(1,13))
prediction = prediction.assign_coords(lon = np.arange(0,360,10))
prediction = prediction.to_dataset(name = 'SIE')

print('Proxies used are:')
print(list(X.keys()))

filedirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfiles\1903Prediction.nc"
prediction.to_netcdf(filedirectory)