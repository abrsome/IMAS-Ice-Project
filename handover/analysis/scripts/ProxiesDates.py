
"""
Proxies start end years.
"""

import numpy as np
import xarray as xr
import pandas as pd

folder = '~/Desktop/IMAS/ncfiles/'  #filepath
xfile = 'CombinedProxies.nc'  #proxy filename

X = xr.open_dataset(folder+xfile)
proxiesnames = list(X.keys())
starts = np.zeros(len(proxiesnames))
ends = np.zeros(len(proxiesnames))

for i in range(len(proxiesnames)):
    t = X[proxiesnames[i]].dropna('year').get_index('year')
    starts[i] = t[0] 
    ends[i] = t[len(t)-1]
    
proxiesdates = pd.DataFrame(list(zip(proxiesnames,starts,ends)),
                            columns = ['Proxy','Start Year','End Year'])
proxiesdates = proxiesdates.sort_values(by = 'Start Year').reset_index(drop=True)

print(proxiesdates)
print("==========================================================")

startyear = 1903
for i in np.arange(1995,2021):
    endyear = i
    yvar = 'SIE'
    
    folder = '~/Desktop/IMAS/ncfiles/'  #filepath
    pfile = 'Proxy_combined_data_v4.nc'  #proxy filename
    xfile = 'CombinedProxies.nc'  #proxy filename
    df = xr.open_dataset(folder+pfile).sel(year= slice(startyear, endyear))
    X = xr.open_dataset(folder+xfile).sel(year= slice(startyear, endyear))
 
    X = X.to_dataframe().dropna(axis=1).to_xarray()
    print("Predictors covering {} to {}: {}".format(startyear, i,len(X.data_vars)))