import xarray as xr
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import os
import bisect
import matplotlib.pyplot as plt
"""
Producing the IC12 netcdf4 file from the excel data.
"""
# -----------------------
rawfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfilesraw"
outfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfiles"
os.chdir(rawfiledirectory)
filename = 'Datasets_IC12_FK17_TIR18_SarahWauthy.xlsx'
sheet = 'IC12'
# -----------------------
#Read in the chemistry data to an ndarray
chemistry = pd.read_excel(filename, sheet_name = sheet, 
                          header=3, usecols = "E:J").to_numpy()
chemistry = chemistry[~np.isnan(chemistry).all(axis=1), :]  #remove empty rows
#Manual delete row 773 as there seems to be an incorrect record at 780 in xlsx
#and it messes up the operations ahead if not removed
if(sheet == 'IC12'):
    chemistry = np.delete(chemistry,773,axis=0)
# -----------------------
#Get the agemodel depths and years into separate arrays (easier for bisect)
agedepths = pd.read_excel(filename, sheet_name = sheet, 
                          header=3, usecols = "A", nrows=270).squeeze()
ageyears = pd.read_excel(filename, sheet_name = sheet, 
                         header=3, usecols = "B", nrows=270).squeeze()

#Stack a zeros column (index 6) alongside the chemistry data for a year dim
z = np.zeros((len(chemistry),1), dtype=int)
chemistry = np.append(chemistry, z, axis=1)
# =========================
"""
Records which cross a depth-year boundary in the age model are split into two
records with the boundary depth taking the top/bottom depth in either split,
with all chemistry data being copied into both.
Bisect returns the index for this depth in agedepths, giving the following
index if equal to a depth-year boundary.
"""
for i in range(len(chemistry)): #1142
    # Get agemodel indices for top and bottom depths
    if(not(pd.isna(chemistry[i,0]))):
        t = bisect.bisect(agedepths, chemistry[i,0]) 
        b = bisect.bisect(agedepths, chemistry[i,1])
        chemistry[i,6] = ageyears[t]
        if(t != b):
            split = chemistry[i] #duplicate the record that crosses a boundary
            #set the top of the split sample to be the boundary, append
            split[0] = agedepths[bisect.bisect(agedepths, chemistry[i,0])]
            split[6] = ageyears[b] #Set new year
            chemistry = np.vstack([chemistry, split])
            
            #set bottom of original sample to be the boundary
            chemistry[i,1] = agedepths[bisect.bisect(agedepths,
                                                     chemistry[i,0])] 
# -----------------------            
#Sort appended array asc in topdepth
chemistry = chemistry[chemistry[:, 0].argsort()]
# =========================          
#Creating netCDF
os.chdir(outfiledirectory)
ncout = Dataset('IC12.nc','w','NETCDF4') # using netCDF3 for output format 
ncout.description = "IC12 ice core chemical records, published in Philippe et al., 2016. IC12 position: -70.234690S, 26.334900E"

ncout.createDimension('topdepth',size=None)
yearvar = ncout.createVariable('year','int',('topdepth'))
yearvar[:] = chemistry[:,6]
tdvar = ncout.createVariable('topdepth','float32',('topdepth'))
tdvar[:] = chemistry[:,0]
thickvar = ncout.createVariable('thickness', 'float32',('topdepth'),fill_value=np.nan)
thickvar.description = "Thickness of this sample"
navar = ncout.createVariable('Na', 'float32',('topdepth'),fill_value=np.nan)
navar.units = "ppb"
msavar = ncout.createVariable('MSA', 'float32',('topdepth'),fill_value=np.nan)
msavar.units = "ppb"
so4var = ncout.createVariable('SO4', 'float32',('topdepth'),fill_value=np.nan)
so4var.units = "ppb"

for i in range(len(chemistry)):
    thickvar[i] = chemistry[i,1]-chemistry[i,0]
    navar[i] = chemistry[i,3]    
    msavar[i] = chemistry[i,4]
    so4var[i] = chemistry[i,5]
    
ncout.close()
