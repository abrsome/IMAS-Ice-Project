
"""
Producing the netcdf4s for the three MBS datasheets.
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
filename = 'MBS data.xlsx'

sheets = ['MBS Main', 'MBS Charlie', 'MBS Alpha']
outnames = ['MBS_Main.nc', 'MBS_Charlie.nc', 'MBS_Alpha.nc']

for s in range(3):
    sheet = sheets[s]
    
    #Change to dir with raw data
    os.chdir(rawfiledirectory)    
    #Read in the chemistry data to an ndarray
    data = pd.read_excel(filename, sheet_name = sheet,
                              header=None, usecols = "A:F")
    
    #Rename dating as year, then swap year to col 0
    data[2][0] = 'year'
    data[0], data[2] = data[2], data[0]
    
    #Drop rows w no year
    data = data.dropna(subset=[0])
    data = data.to_numpy()
    
    names = data[0]
    data = data[1:].transpose()
    
    #For some reason the ASCII in the names from the excel files causes
    #the netcdf4 module to import the variable as a group instead of a var
    #seems like an odd bug, but can be fixed by renaming the var names
    names[3] = 'MSA'; names[4] = 'SO4'; names[5] = 'Na'
    
    #==============================================================================
    #Output netcdf file
    
    os.chdir(outfiledirectory)
    
    ncout = Dataset(outnames[s],'w','NETCDF4')
    if sheet == 'MBS Main':
        ncout.description = "Data for " + sheet + "from Christopher Plummer. Year taken as floor value from xlsx and dropped when nan. Note: MSA below 20 metres suffered from losses during storage - was analysed from a different core piece"
    else:
        ncout.description = "Data for " + sheet + "from Christopher Plummer. Year taken as floor value from xlsx and dropped when nan."
        
    ncout.createDimension('year',size=None)
    yearvar = ncout.createVariable('year','int',('year'))
    yearvar[:] = data[0]
    yearvar[:] = list(map(math.floor,data[0]))
    
    for i in np.arange(1,len(names)):
        if i == 2: #string for sample name
            attr = ncout.createVariable(names[i],'str',('year'))
        else:
            attr = ncout.createVariable(names[i],'float32',('year'))
        attr[:] = data[i]
    
    ncout.close()

