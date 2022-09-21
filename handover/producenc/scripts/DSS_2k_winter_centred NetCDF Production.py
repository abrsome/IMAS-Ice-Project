
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import os
import math
"""
Producing the netcdf4 for DSS_2k_winter_centred.
"""
#==============================================================================
#Load data, set globals
rawfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfilesraw"
outfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfiles"
os.chdir(rawfiledirectory)
filename = 'DSS_2k_winter_centred.csv'
sheet = 'DSS_2k_winter_centred'

#Read in the chemistry data to an ndarray
data = pd.read_csv(filename, header=0, usecols = np.arange(1,12,1)).to_numpy()


#==============================================================================
#Output netcdf file
os.chdir(outfiledirectory)
ncout = Dataset('DSS_2k_winter_centred.nc','w','NETCDF4') # using netCDF3 for output format 
ncout.description = "DSS_2k_winter_centred records."

ncout.createDimension('year',size=None)
yearvar = ncout.createVariable('year','int',('year'))
yearvar[:] = list(map(math.floor,data[:,0]))
#Need to set all units in #
sodium = ncout.createVariable('sodium','float32',('year'))
sodium[:] = data[:,1]
#
chloride = ncout.createVariable('chloride','float32',('year'))
chloride[:] = data[:,2]
#
magnesium = ncout.createVariable('magnesium','float32',('year'))
magnesium[:] = data[:,3]
#
sulphate = ncout.createVariable('sulphate','float32',('year'))
sulphate[:] = data[:,4]
#
d180 = ncout.createVariable('d180','float32',('year'))
d180[:] = data[:,5]
#
layer_thickness = ncout.createVariable('layer_thickness','float32',('year'))
layer_thickness[:] = data[:,6]
#
accumulation_rate = ncout.createVariable('accumulation_rate','float32',('year'))
accumulation_rate[:] = data[:,7]
#
DJFMAM = ncout.createVariable('DJFMAM','float32',('year'))
DJFMAM[:] = data[:,8]
#
JJASON = ncout.createVariable('JJASON','float32',('year'))
JJASON[:] = data[:,9]
#
DJFM = ncout.createVariable('DJFM','float32',('year'))
DJFM[:] = data[:,10]
#

ncout.close()
