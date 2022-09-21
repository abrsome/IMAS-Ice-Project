
"""
Producing combined proxy dataset with added proxies.
"""

#==============================================================================
#Imports, globals, filepaths, loading data

import xarray as xr

folder = '~/Desktop/IMAS/ncfiles/'  #filepath
pfile = 'Proxy_combined_data_v4.nc'  #proxy filename
df = xr.open_dataset(folder+pfile) #.sel(year= slice(startyear, endyear))

#==============================================================================
#Proxies data into time series

#Mean on month as WHG_dex has monthly values
X = df[(list(df.keys())[1:])].mean('month')

#------------------------------------------------------------------------------
#Merge in DSS data
s = xr.open_dataset(folder+'DSS_2k_summer_centred.nc')
keys = list(s.keys())
renamed = list(map(lambda x: 'DSS_summer_'+x, keys))
s = s.rename(dict(zip(keys,renamed)))
X = X.merge(s)

s = xr.open_dataset(folder+'DSS_2k_winter_centred.nc')
keys = list(s.keys())
renamed = list(map(lambda x: 'DSS_winter_'+x, keys))
s = s.rename(dict(zip(keys,renamed)))
X = X.merge(s)

#------------------------------------------------------------------------------
#Merge in Sarah's ice core data
#Need to calculate avg for each year by doing weighted avg from different
#thickness sections in each year
cores = ['TIR18', 'FK17','IC12']
proxies = ['MSA','Na','SO4']
for c in range(3): #3 cores
    cdata = xr.open_dataset(folder+cores[c]+'.nc')        
    for p in range(3): #3 proxies
        proxy = proxies[p]
    
        # Do thickness weighting for the proxy.
        for i in range(len(cdata['topdepth'])):
            cdata[proxy][i] = cdata['thickness'][i]*cdata[proxy][i]
        
        cdata = cdata.rename({proxy: cores[c]+'_'+proxy})
    cdata = cdata.groupby("year").mean()
    cdata = cdata.drop_vars('thickness')
    #Due to cutoff in age model, remove first year as that contains
    # all values beyond the age model and so is incorrect
    cdata = cdata.drop_sel(year=cdata.get_index('year')[0])
    X = X.merge(cdata)

#------------------------------------------------------------------------------
#Merge in MBS Data
mbsdata = ['MBS_Main','MBS_Alpha','MBS_Charlie']
for c in range(3):
    ds = mbsdata[c]
    s = xr.open_dataset(folder+ds+'.nc')    
    s = s.drop_vars(['Mid Depth','Sample name'])
    keys = list(s.keys())
    renamed = list(map(lambda x: ds+'_'+x, keys))
    s = s.rename(dict(zip(keys,renamed))).groupby('year').mean()
    X = X.merge(s)

#------------------------------------------------------------------------------
#Merge in Crockart et al data
X = X.merge(xr.open_dataset(folder+'Crockart_et_al_2_0.nc'))

#------------------------------------------------------------------------------
CombinedProxies = X
filedirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfiles\CombinedProxies.nc"
CombinedProxies.to_netcdf(filedirectory)