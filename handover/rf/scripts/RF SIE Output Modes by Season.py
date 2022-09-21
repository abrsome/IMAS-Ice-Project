
"""
Testing different output modes using seasons not single months.
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
md = None #max depth
rs = 0 #random state
np.random.seed(rs)

predstartyear = 1979 #start of predicted period
startyear = 1979
endyear = 1995

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
    2. Use a scaler to interpolate nan values.
"""
#1. To drop predictors containing NaN in this range:
#Doesn't look like xarray has equivalent of dropna(axis=1) so I am going
#through pandas and back to xarray:
X = X.to_dataframe().dropna(axis=1).to_xarray()

#mode of prediction, respective to array below
outputmodes = 4 #number of output modes
outputname = ['Individual','Month','Longitudinal','Year']
"""
Month = consider data from all longitudes in a month as one target
Longitudinal = consider data from all months in that longitude as one target
Year = consider the entire year's data as a target variable
"""

#==============================================================================
#Preparing data

#Global SIE mean
mean_sie = df['SIE'].mean('month').mean('lon').mean('year')
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

#Standardise values of proxies
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#------------------------------------------------------------------------------
#Get SIE into bins of 10degree longitude sections
y = df[yvar].groupby_bins('lon', np.arange(0,361,10)).mean()

"""
Create a seasonal dataset. Instead of using 12 individual months,
use 10 3-month seasons. Only using 10 currently to skip work of wrap around
seasons on calendar years (DFJ, NDJ), but this can be fixed later.
Indexing works as follows: 0=JFM, 1= FMA, ..., 9=OND
"""
seasonal = np.empty([10, y.shape[0],36])
for m in range(10):
    m+=1
    months = [m, m+1, m+2]
    seasonal[m-1] = y.sel(month = months).mean('month')

#Change indexing of seasonal to be [year, season, longitude]
seasonal = seasonal.transpose([1,0,2])
#Form xarray version
sxr = xr.DataArray(seasonal)
dicti = {'dim_0': 'year', 'dim_1': 'season', 'dim_2': 'lon'}
sxr = sxr.rename(dicti)
sxr = sxr.assign_coords(year=(np.arange(startyear,endyear+1)),
                        lon=(np.arange(10,361,10)),
                        season=(np.arange(1,11)))


#==============================================================================
#Training and testing model

#Array to store final 2d error arrays from each mode
errorarrays = []


for mode in range(outputmodes):
    #Array to store root mean square error as percentage of mean
    #RMSEPOM = Root Mean Squared Error as Percentage Of Mean
    rmsepom = np.zeros([10,36]) #Initialise an array of correlations
    
    if(mode == 0): #Individual = consider each month/lon pair separately
        for s in range(10): #loop over the months
            s += 1 #month index in xarray starts at 1
            monthdata = sxr.sel(season=s)
            for long in range(36): 
                longend = (long+1)*10 #end of range of longitude section
                monthlondata = monthdata.sel(lon=longend)
                y_train = monthlondata.sel(year = trainyears).to_numpy()
                y_test = monthlondata.sel(year = testyears).to_numpy()
                
                #Training model
                regressor = RandomForestRegressor(n_estimators = n_est,
                                                  random_state = rs, max_depth = md)
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
                
                #Put error into heatmap
                rmsepom[s-1,int((longend-10)/10)] = 100*rmse/mean_sie
        errorarrays.append(rmsepom) #Add in error array for this mode
        
    elif(mode == 1): #Month = consider data from all longitudes in a month as one target
        for s in range(10): #loop over the months
            s += 1 #month index in xarray starts at 1
            monthdata = sxr.sel(season=s)
            y_train = monthdata.sel(year = trainyears).to_numpy()
            y_test = monthdata.sel(year = testyears).to_numpy()
            
            #Training model
            regressor = RandomForestRegressor(n_estimators = n_est, 
                                                  random_state = rs, max_depth = md)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            
            #Convert nested arrays from longitudes to year time series
            y_test = y_test.transpose()
            y_pred = y_pred.transpose() 
            
            for long in range(36):
                rmse = np.sqrt(metrics.mean_squared_error(y_test[long], y_pred[long]))
                rmsepom[s-1,long] = 100*rmse/mean_sie
                
        errorarrays.append(rmsepom) #Add in error array for this mode
        
    elif(mode == 2): #Longitudinal = consider data from all months in that longitude as one target
        for long in range(36):
            longend = (long+1)*10
            longdata = sxr.sel(lon = longend)
            y_train = longdata.sel(year = trainyears).to_numpy()
            y_test = longdata.sel(year = testyears).to_numpy()
            
            #Training model
            regressor = RandomForestRegressor(n_estimators = n_est, 
                                                  random_state = rs, max_depth = md)
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            
            #Convert nested arrays from months to year time series
            y_test = y_test.transpose()
            y_pred = y_pred.transpose() 
            
            for s in range(10):
                rmse = np.sqrt(metrics.mean_squared_error(y_test[s], y_pred[s]))
                rmsepom[s, long] = 100*rmse/mean_sie
        
        errorarrays.append(rmsepom) #Add in error array for this mode
    
    elif(mode == 3): #Year = consider the entire year's data as a target variable
        y_train = sxr.sel(year = trainyears).to_numpy()
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1] * y_train.shape[2]) # 360 entries per year
        y_test = y.sel(year = testyears).to_numpy()
        
        #Training model
        regressor = RandomForestRegressor(n_estimators = n_est, 
                                                  random_state = rs, max_depth = md)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        
        #Transform pred and test into standard format to get time series
        y_pred = y_pred.reshape(y_pred.shape[0],10,36).transpose()
        y_test = y_test.transpose()
        
        
        for l in range(36):
            for s in range(10):
                rmse = np.sqrt(metrics.mean_squared_error(y_test[l,s], y_pred[l,s]))
                rmsepom[s,l] = 100*rmse/mean_sie   
                
        errorarrays.append(rmsepom) #Add in error array for this mode                
        
#==============================================================================
#draw function for heatmap analysis
def draw(rmsepom):
    ax = plt.subplot()
    
    rmsepomxr = xr.DataArray(rmsepom)
    rmsepomxr.plot.pcolormesh(levels = clev)
    
    ax.set_title('Error in '+yvar+' predictions using RFs')
    ax.set_ylabel('Month')
    ax.set_xlabel('Longitude')
    ax.set_xticks(ticks=np.arange(3,36,3), labels=['','60E','','120E','','180','','120W','','60W',''])
    ax.set_yticks(ticks=np.arange(0,10,1), labels=['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND'], rotation=0)
    ax.annotate("Mode = {}".format(outputname[mode]), xy = (0,0), xytext = (-6,-2.1))
    
    #Seasonal axis read downwards
    #Annotate how to read seasonal axis
    
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
#Analysis

for mode in range(outputmodes):
    rmsepom = errorarrays[mode]

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
      