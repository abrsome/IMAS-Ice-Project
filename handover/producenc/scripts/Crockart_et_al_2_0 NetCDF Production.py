
"""
Producing the netcdf4 for Crockart_et_al_2_0.
"""

#==============================================================================
#imports, load data, set globals
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import os
import math

rawfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfilesraw"
outfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfiles"
os.chdir(rawfiledirectory)
filename = 'Data for Crockart et al. 2.0.xlsx'
sheet = 'Data for Crockart et al.'

#Read in the chemistry data to an ndarray
data = pd.read_excel(filename, sheet_name = sheet, nrows=43,
                          header=None, usecols = "A:O").to_numpy()

names = data[0]

data = data[1:].transpose()

#==============================================================================
#Output netcdf file
os.chdir(outfiledirectory)

ncout = Dataset('Crockart_et_al_2_0.nc','w','NETCDF4')
ncout.description = "Data for Crockart et al 2.0 from https://researchdata.edu.au/el-nio-southern-brown-south/1597740 This is the data for three Mount Brown South (MBS) ice core records (Alpha, Charlie and Main) collected in summer 2017/2018 from East Anatrctica; Alpha, Charlie and Main. And an updated Law Dome (LD, Dome Summit South site) ice core record collected in 2016/2017 (extends the record presented in Vance et al. 2013). Log means log-transformed. Na means sodium, and Cl means chloride (originally both in μEq l-1, although the log-transformed values are presented here). Accum means snowfall accumulation measured in m yr-1 IE (IE stands for ice equivalent). MBS Charlie chloride and soidum values are excluded in the year 1987 as this was an outlier. These are not detrended values."

ncout.createDimension('year',size=None)
yearvar = ncout.createVariable('year','int',('year'))
yearvar[:] = data[0]

for i in np.arange(1,len(names)):
    attr = ncout.createVariable(names[i],'float32',('year'))
    attr[:] = data[i]

ncout.close()
