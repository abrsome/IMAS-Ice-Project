
import xarray as xr
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import os
import bisect
import math
"""
Producing the netcdf4 for DSS_2k_summer_centred.
"""
#==============================================================================
#Load data, set globals
rawfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfilesraw"
outfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfiles"
os.chdir(rawfiledirectory)
filename = 'DSS_2k_summer_centred.csv'
sheet = 'DSS_2k_summer_centred'

#Read in the chemistry data to an ndarray
data = pd.read_csv(filename, header=0, usecols = [1,2,3]).to_numpy()


#==============================================================================
#Output netcdf file
os.chdir(outfiledirectory)
ncout = Dataset('DSS_2k_summer_centred.nc','w','NETCDF4') # using netCDF3 for output format 
ncout.description = "DSS_2k_summer_centred records. Years are rounded down from 2016.5 to 2016."

ncout.createDimension('year',size=None)
yearvar = ncout.createVariable('year','int',('year'))
yearvar[:] = list(map(math.floor,data[:,0]))
nitratevar = ncout.createVariable('nitrate','float32',('year'))
nitratevar[:] = data[:,1]
#nitratevar.units = ""
nsssulphatevar = ncout.createVariable('non_sea_salt_sulphate','float32',('year')) 
nsssulphatevar[:] = data[:,2]
#nsssulphatevar.units = ""
    
ncout.close()
