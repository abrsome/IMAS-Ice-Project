
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import os
import bisect
"""
Producing the TIR18 netcdf4 file from the excel data.
"""
# -----------------------
rawfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfilesraw"
outfiledirectory = r"C:\Users\Alfie\Desktop\IMAS\ncfiles"
os.chdir(rawfiledirectory)
filename = 'Datasets_IC12_FK17_TIR18_SarahWauthy.xlsx'
sheet = 'TIR18'
# -----------------------
#Read in the chemistry data to an ndarray
chemistry = pd.read_excel(filename, sheet_name = sheet, 
                          header=3, usecols = "D:J").to_numpy()
# -----------------------
#Get the agemodel depths and years into separate arrays (easier for bisect)
agedepths = pd.read_excel(filename, sheet_name = sheet, 
                          header=3, usecols = "A", nrows=48).squeeze()
ageyears = pd.read_excel(filename, sheet_name = sheet, 
                         header=3, usecols = "B", nrows=48).squeeze()
#Sample depths actually extend beyond last value in agemodel, so
#this nan must be filled with a value for bisect to work
agedepths[47] = 100 ; ageyears[47] = 1969

#Stack a zeros column (index 6) alongside the chemistry data for a year dim
z = np.zeros((len(chemistry),1), dtype=int)
chemistry = np.append(chemistry, z, axis=1)
#Roll sample id to be in index 7
chemistry = np.roll(chemistry, -1, 1)
# =========================
"""
Records which cross a depth-year boundary in the age model are split into two
records with the boundary depth taking the top/bottom depth in either split,
with all chemistry data being copied into both.
Bisect returns the index for this depth in agedepths, giving the following
index if equal to a depth-year boundary.
"""
for i in range(len(chemistry)):
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
os.chdir(outfiledirectory)
ncout = Dataset('TIR18.nc','w','NETCDF4') # using netCDF3 for output format 
ncout.description = "Preliminary TIR18 data, agemodel unconfirmed. TIR18 position: -70.499600S, 21.880170E"

ncout.createDimension('topdepth',size=None)
yearvar = ncout.createVariable('year','int',('topdepth'))
yearvar[:] = chemistry[:,6]
tdvar = ncout.createVariable('topdepth','float32',('topdepth'))
tdvar[:] = chemistry[:,0]
thickvar = ncout.createVariable('thickness', 'float32',('topdepth'),fill_value=np.nan)
thickvar.description = "Thickness of this sample"
navar = ncout.createVariable('Na', 'float32',('topdepth'),fill_value=np.nan)
navar.units = "ppb"
navar[:] = chemistry[:,3]
msavar = ncout.createVariable('MSA', 'float32',('topdepth'),fill_value=np.nan)
msavar.units = "ppb"
msavar[:] = chemistry[:,4]
so4var = ncout.createVariable('SO4', 'float32',('topdepth'),fill_value=np.nan)
so4var.units = "ppb"
so4var[:] = chemistry[:,5]
sampleidvar = ncout.createVariable('sample_id','str',('topdepth'))
sampleidvar[:] = chemistry[:,7]

for i in range(len(chemistry)):
    thickvar[i] = chemistry[i,1]-chemistry[i,0]    
ncout.close()
