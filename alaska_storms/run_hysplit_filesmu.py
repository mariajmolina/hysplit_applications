import pysplit
import pandas as pd
import numpy as np
from wrf import getvar, get_basemap, latlon_coords
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import matplotlib as mpl
import xarray as xr
from itertools import product

###############################################################

parser = argparse.ArgumentParser(description='Calculating hysplit files.')

parser.add_argument("--climate", choices=["hist", "pgw"], required=True, type=str, help="This is the hist or pgw choice.")
parser.add_argument("--month", choices=["may", "jun", "jul", "aug"], required=True, type=str, help="Storm month as three letter string.")
parser.add_argument("--subregion", required=True, type=str, help="The subregion + number. There is no subregion2")

args=parser.parse_args()

###############################################################

which_climate=args.climate
which_month=args.month
which_region=args.subregion

###############################################################

def get_winddir(comp_brngs):
    md = 270 - comp_brngs
    if md < 0:
        new_md = md + 360
    else:
        new_md = md
    return new_md 

def nan_if(arr, value):
    return np.where(arr == value, np.nan, arr)

###############################################################

wrffile = Dataset('/glade/scratch/molina/basile/wrfout_d01_2002-12-30_18:00:00.nc4')
tfile = getvar(wrffile, 'tk')
m = get_basemap(tfile)
lats, lons = latlon_coords(tfile)

grid_res = 50
xMax,yMax = m(m.urcrnrlon, m.urcrnrlat) 
xMin,yMin = m(m.llcrnrlon, m.llcrnrlat)         
x_range = (xMax-xMin) / 1000 
y_range = (yMax-yMin) / 1000 
numXGrids = round(x_range / grid_res + .5,0) 
numYGrids = round(y_range / grid_res + .5,0)        
xi = np.linspace(xMin, xMax, int(numXGrids))
yi = np.linspace(yMin, yMax, int(numYGrids))

###############################################################

#trajgroup = pysplit.make_trajectorygroup(
#    f'/glade/scratch/molina/basile/{which_climate}_traj_{which_month}/trajid*_{which_region}_ens*_{which_month}*summer*')
trajgroup = pysplit.make_trajectorygroup(
    f'/glade/scratch/molina/basile/{which_climate}_traj_{which_month}/trajid*_{which_region}_ens*_{which_month}*spring*')
#print(trajgroup.trajcount)

###############################################################

bad_keyerror = []
for trajnum, traj in enumerate(trajgroup):
    try:
        print(traj.trajid)
        traj.moisture_uptake(precipitation=-0.2, evaporation=0.2, interval=6, vlim='pbl')
        print('Uptake calculation for', trajnum+1, 'underway of', trajgroup.trajcount)
    except KeyError:
        bad_keyerror.append(traj.trajid)
    except TypeError:
        bad_keyerror.append(traj.trajid)
    except ValueError:
        bad_keyerror.append(traj.trajid)

if len(bad_keyerror) > 0:
    trajgroup.pop(trajid=bad_keyerror)

###############################################################

dq_all = np.array([traj.uptake.dq_initial.values[j] for traj in trajgroup for j,i in enumerate(traj.uptake.below.values)]).flatten()
dq_all = np.array([0.0 if v is None else v for v in dq_all]).flatten()
q_all = np.array([traj.uptake.q.values[j] for traj in trajgroup for j,i in enumerate(traj.uptake.below.values)]).flatten()   
x_dqall = np.array([(traj.uptake.geometry.apply(lambda p: p.x).values[j]) for traj in trajgroup for j,i in enumerate(traj.uptake.below.values)]).flatten()
y_dqall = np.array([(traj.uptake.geometry.apply(lambda p: p.y).values[j]) for traj in trajgroup for j,i in enumerate(traj.uptake.below.values)]).flatten()
x_dqalls, y_dqalls = m(x_dqall, y_dqall)

dq_below = np.array([traj.uptake.dq.values[j] for traj in trajgroup for j,i in enumerate(traj.uptake.below.values) if i]).flatten() 
timestep_below = np.array([traj.uptake.Timestep.values[j] for traj in trajgroup for j,i in enumerate(traj.uptake.below.values) if i]).flatten() 
traj_uptake_below = np.array([traj.uptake.below.values[j] for traj in trajgroup for j,i in enumerate(traj.uptake.below.values) if i]).flatten()
x_uptake = np.array([(traj.uptake.geometry.apply(lambda p: p.x).values[j]) for traj in trajgroup for j,i in enumerate(traj.uptake.below.values) if i]).flatten()
y_uptake = np.array([(traj.uptake.geometry.apply(lambda p: p.y).values[j]) for traj in trajgroup for j,i in enumerate(traj.uptake.below.values) if i]).flatten()
z_uptake = np.array([(traj.uptake.geometry.apply(lambda p: p.z).values[j]) for traj in trajgroup for j,i in enumerate(traj.uptake.below.values) if i]).flatten()
q_below = np.array([traj.uptake.q.values[j] for traj in trajgroup for j,i in enumerate(traj.uptake.below.values) if i]).flatten()       
x_uptakes, y_uptakes = m(x_uptake, y_uptake)

###############################################################

#freqs
mfreq_2d, _, _ = np.histogram2d(y_uptakes, x_uptakes, bins=(yi, xi))
dqallfreq_2d, _, _ = np.histogram2d(y_dqalls, x_dqalls, bins=(yi, xi))

#variables
dq_all_2d, _, _ = np.histogram2d(y_dqalls, x_dqalls, bins=(yi, xi), weights=dq_all, normed=False)
q_all_2d, _, _ = np.histogram2d(y_dqalls, x_dqalls, bins=(yi, xi), weights=q_all, normed=False)
dq_below_2d, _, _ = np.histogram2d(y_uptakes, x_uptakes, bins=(yi, xi), weights=dq_below, normed=False)
q_below_2d, _, _ = np.histogram2d(y_uptakes, x_uptakes, bins=(yi, xi), weights=q_below, normed=False)
dq_below_time, _, _ = np.histogram2d(y_uptakes, x_uptakes, bins=(yi, xi), weights=timestep_below, normed=False)
traj_uptake_below_2d, _, _ = np.histogram2d(y_uptakes, x_uptakes, bins=(yi, xi), weights=traj_uptake_below, normed=False)
z_uptake_2d, _, _ = np.histogram2d(y_uptakes, x_uptakes, bins=(yi, xi), weights=z_uptake, normed=False)

###############################################################

dq_all_grid = np.zeros(dq_all_2d.shape)
q_all_grid = np.zeros(q_all_2d.shape)
dq_below_grid = np.zeros(dq_below_2d.shape)
q_below_grid = np.zeros(q_below_2d.shape)
dq_below_time_grid = np.zeros(dq_below_time.shape)
traj_uptake_grid = np.zeros(traj_uptake_below_2d.shape)
z_uptake_grid = np.zeros(z_uptake_2d.shape)

###############################################################

for i, j in product(range(len(mfreq_2d[:,0])),range(len(mfreq_2d[0,:]))):

    #dq_all
    if np.isfinite(dq_all_2d[i,j]):
        if dq_all_2d[i,j] != 0.0:
            dq_all_grid[i,j] = dq_all_2d[i,j]/dqallfreq_2d[i,j]
        else:
            dq_all_grid[i,j] = dq_all_2d[i,j]
    else:
        dq_all_grid[i,j] = dq_all_2d[i,j]
        
    #q_all
    if np.isfinite(q_all_2d[i,j]):
        if q_all_2d[i,j] != 0.0:
            q_all_grid[i,j] = q_all_2d[i,j]/dqallfreq_2d[i,j]
        else:
            q_all_grid[i,j] = q_all_2d[i,j]
    else:
        q_all_grid[i,j] = q_all_2d[i,j]

    #dq_below
    if np.isfinite(dq_below_2d[i,j]):
        if dq_below_2d[i,j] != 0.0:
            dq_below_grid[i,j] = dq_below_2d[i,j]/mfreq_2d[i,j]
        else:
            dq_below_grid[i,j] = dq_below_2d[i,j]
    else:
        dq_below_grid[i,j] = dq_below_2d[i,j]
        
    #q_below
    if np.isfinite(dq_below_2d[i,j]):
        if dq_below_2d[i,j] != 0.0:
            dq_below_grid[i,j] = dq_below_2d[i,j]/mfreq_2d[i,j]
        else:
            dq_below_grid[i,j] = dq_below_2d[i,j]
    else:
        dq_below_grid[i,j] = dq_below_2d[i,j]

    #q_below_time
    if np.isfinite(dq_below_time[i,j]):
        if dq_below_time[i,j] != 0.0:
            dq_below_time_grid[i,j] = dq_below_time[i,j]/mfreq_2d[i,j]
        else:
            dq_below_time_grid[i,j] = dq_below_time[i,j]
    else:
        dq_below_time_grid[i,j] = dq_below_time[i,j]

    #uptake_frac_below
    if np.isfinite(traj_uptake_below_2d[i,j]):
        if traj_uptake_below_2d[i,j] != 0.0:
            traj_uptake_grid[i,j] = traj_uptake_below_2d[i,j]/mfreq_2d[i,j]
        else:
            traj_uptake_grid[i,j] = traj_uptake_below_2d[i,j]
    else:
        traj_uptake_grid[i,j] = traj_uptake_below_2d[i,j]                

    #z_uptake
    if np.isfinite(z_uptake_2d[i,j]):
        if z_uptake_2d[i,j] != 0.0:
            z_uptake_grid[i,j] = z_uptake_2d[i,j]/mfreq_2d[i,j]
        else:
            z_uptake_grid[i,j] = z_uptake_2d[i,j]
    else:
        z_uptake_grid[i,j] = z_uptake_2d[i,j] 
        
###############################################################

y = np.zeros(len(yi)-1)
x = np.zeros(len(xi)-1)
for a in range(len(y)):
    y[a] = np.nanmean([yi[a],yi[a+1]])
for b in range(len(x)):
    x[b] = np.nanmean([xi[b],xi[b+1]])

###############################################################
    
ds = xr.Dataset({
                 'moisture_frequency':(['lats','lons'],mfreq_2d),
                 'dqall_frequency':(['lats','lons'],dqallfreq_2d),
                 'dq_all':(['lats','lons'],dq_all_grid),
                 'q_all':(['lats','lons'],q_all_grid),
                 'dq_below':(['lats','lons'],dq_below_grid),
                 'q_below':(['lats','lons'],q_below_grid),
                 'dq_below_uptake_grid':(['lats','lons'],dq_below_time_grid),
                 'traj_uptake_below':(['lats','lons'],traj_uptake_grid),
                 'z_uptake_grid':(['lats','lons'],z_uptake_grid)
                },
                 coords={'lats':(['lats'],y),
                         'lons':(['lons'],x)},
                 attrs={'Trajectories and File Author':'Maria J. Molina',
                        'Trajectory Count':str(trajgroup.trajcount)})

###############################################################

ds.to_netcdf(f'/glade/scratch/molina/basile/traj_files/moisturefile_{which_month}_{which_region}_{which_climate}.nc')

###############################################################
