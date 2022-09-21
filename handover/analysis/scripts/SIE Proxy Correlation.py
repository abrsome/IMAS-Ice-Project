"""
For all proxies by only year dimension, WHG dxs is an annual mean.
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from datetime import datetime

start = datetime.now()

seasons = ['OND','NDJ','DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND']

#input file info
folder = '~/Desktop/IMAS/ncfiles/'  #filepath
pfile = 'Proxy_combined_data_v4.nc'  #proxy filename
pvar = 'SIE'  #proxy sea ice extent variable name
pdata = xr.open_dataset(folder+pfile) #open proxy as Dataset

corrs = np.zeros([12,36]) #Initialise an array of correlations
startyear = 1979  #first year of SIE, fixed as later than all proxies
#Arrays of proxies + respective end years
#SIE ends after all proxies so they determine end yr
proxies = ['AP_stacked_MSA', 'Law_Dome_MSA', 'Ferrigno_MSA','WHG_dex','SOFI_formation','SOFI_breakout','SOFI_duration']
endyears = [2002,1995,2010,2006,2008,2008,2008]  #respective end years of proxies

for i in range(7):
    
    proxy = proxies[i]
    endyear = endyears[i]
    
    if proxy=='WHG_dex':
        pts = pdata[proxy].sel(year = slice(startyear,endyear))
        pts = pts.mean('month')
    else:
        pts = pdata[proxy].sel(year = slice(startyear,endyear)) 

    
    #Creating time series over months
    for m in range(12): #loop over the months
        m+=1
        months = [(m-3)%12+1, (m-2)%12+1, (m-1)%12+1] #nonneg month ints for season ending in m
        monat = pdata[pvar].sel(year = slice(startyear,endyear), month=months) 
        monat = monat.mean('month')
        for l in np.arange(0.25,360,10): #loop over the longitudes
            sie = monat.sel(lon= slice(l, l+10)) 
            sie = sie.mean(dim='lon')
            corrs[m-1,int((l-0.25)/10)] = stats.pearsonr(pts,sie)[0] #Place corcoef in corrs
    
    ax = plt.subplot()
    
    clev = [-0.8,-0.7,-0.6,-0.5,-0.4,0.4,0.5,0.6,0.7,0.8]  #contour levels
    #clev = np.arange(-1, 1.1, 0.1)
    corrsxr = xr.DataArray(corrs)
    corrsxr.plot.pcolormesh(levels = clev)
    
    ax.set_title(proxy)
    ax.set_ylabel('Season')
    #plt.yticks(ticks=np.arange(12))
    ax.set_xlabel('Longitude')
    ax.set_xticks(ticks=np.arange(3,36,3), labels=['','60E','','120E','','180','','120W','','60W',''])
    ax.set_yticks(ticks=np.arange(-0.5,12,1), labels=seasons, rotation=0)
    
    plt.show()

end = datetime.now()
print("The time of execution of above program is :",
      str(end-start)[5:], "seconds.")
#New cmap
"""
Considering seasonality: Pearson correlation of a time series for a given month
will consider effect of annual changes in dxs. We will not see effect of 
seasonality as correlation coefficients of months are calculated independently.
"""
