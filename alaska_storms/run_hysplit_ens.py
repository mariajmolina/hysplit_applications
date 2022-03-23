import pandas as pd
import datetime
import numpy as np
import argparse
import math
from pysplit.trajectory_generator import generate_bulktraj

############################################################################
############################################################################

parser = argparse.ArgumentParser(description='Calculating hysplit trajs.')

parser.add_argument("--climate", choices=["hist", "pgw"], required=True, type=str, help="This is the hist or pgw choice.")
parser.add_argument("--month", required=True, type=int, help="Storm month for trajectory calculation.")
parser.add_argument("--ens", required=True, type=int, help="Ensemble number.")

args=parser.parse_args()

which_climate=args.climate
which_month=args.month
ens_number=args.ens

############################################################################
############################################################################

work_help1=np.hstack([np.array([0 for i in range(36)]),
                      np.array([1 for i in range(36)]),
                      np.array([2 for i in range(36)]),
                      np.array([3 for i in range(36)]),
                      np.array([4 for i in range(36)]),
                      np.array([5 for i in range(36)]),
                      np.array([6 for i in range(36)]),
                      np.array([7 for i in range(36)]),
                      np.array([8 for i in range(36)]),
                      np.array([9 for i in range(36)])
                     ])

work_help2=np.hstack([np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)]),
                      np.array([i+1 for i in range(36)])
                     ])

which_working=f"work{work_help2[ens_number]}_{work_help1[ens_number]}"

############################################################################
############################################################################

csv_file=pd.read_csv(f'/glade/work/bpoujol/Moisture_tracking/trajectory_information_{which_climate}.csv')

dates=[]
for datetime_string in csv_file['TIME (UTC)'].values:
    dates.append(datetime.datetime.strptime(datetime_string, '%Y-%m-%d_%H:%M:%S'))
ready_dates=pd.to_datetime(np.array(dates))

csv_file['YEAR']=ready_dates.year
csv_file['MONTH']=ready_dates.month
csv_file['DAY']=ready_dates.day
csv_file['HOUR']=ready_dates.hour

csv_file=csv_file[csv_file['MONTH']==which_month]

############################################################################
############################################################################

def ens_create(ens_num, lat, lon):
    
    """Extract the ensemble member's lat and lon coordinates.
    
    """
    ens_help=np.hstack([np.array([i for i in range(49)]),
                        np.array([i for i in range(49)]),
                        np.array([i for i in range(49)]),
                        np.array([i for i in range(49)]),
                        np.array([i for i in range(49)]),
                        np.array([i for i in range(49)]),
                        np.array([i for i in range(49)])
                       ])
    ens_num=ens_help[ens_num]
    if ens_num==0:
        return compute_displacement(lat, lon)
    if ens_num==1:
        return compute_displacement(lat, lon, dist=1, bear=90)
    if ens_num==2:
        return compute_displacement(lat, lon, dist=2, bear=90)
    if ens_num==3:
        return compute_displacement(lat, lon, dist=3, bear=90)
    if ens_num==4:
        return compute_displacement(lat, lon, dist=1, bear=270)
    if ens_num==5:
        return compute_displacement(lat, lon, dist=2, bear=270)
    if ens_num==6:
        return compute_displacement(lat, lon, dist=3, bear=270)
    if ens_num==7:
        return compute_displacement(lat, lon, dist=1, bear=180)
    if ens_num==8:
        return compute_displacement(lat, lon, dist=2, bear=180)
    if ens_num==9:
        return compute_displacement(lat, lon, dist=3, bear=180)
    if ens_num==10:
        return compute_displacement(lat, lon, dist=1, bear=0)
    if ens_num==11:
        return compute_displacement(lat, lon, dist=2, bear=0)
    if ens_num==12:
        return compute_displacement(lat, lon, dist=3, bear=0)
    if ens_num==13:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=0)
        return compute_displacement(newlat, newlon, dist=1, bear=90)
    if ens_num==14:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=0)
        return compute_displacement(newlat, newlon, dist=2, bear=90)
    if ens_num==15:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=0)
        return compute_displacement(newlat, newlon, dist=3, bear=90)
    if ens_num==16:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=0)
        return compute_displacement(newlat, newlon, dist=1, bear=90)
    if ens_num==17:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=0)
        return compute_displacement(newlat, newlon, dist=2, bear=90)
    if ens_num==18:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=0)
        return compute_displacement(newlat, newlon, dist=3, bear=90)
    if ens_num==19:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=0)
        return compute_displacement(newlat, newlon, dist=1, bear=90)
    if ens_num==20:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=0)
        return compute_displacement(newlat, newlon, dist=2, bear=90)
    if ens_num==21:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=0)
        return compute_displacement(newlat, newlon, dist=3, bear=90)
    if ens_num==22:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=0)
        return compute_displacement(newlat, newlon, dist=1, bear=270)
    if ens_num==23:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=0)
        return compute_displacement(newlat, newlon, dist=2, bear=270)
    if ens_num==24:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=0)
        return compute_displacement(newlat, newlon, dist=3, bear=270)
    if ens_num==25:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=0)
        return compute_displacement(newlat, newlon, dist=1, bear=270)
    if ens_num==26:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=0)
        return compute_displacement(newlat, newlon, dist=2, bear=270)
    if ens_num==27:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=0)
        return compute_displacement(newlat, newlon, dist=3, bear=270)
    if ens_num==28:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=0)
        return compute_displacement(newlat, newlon, dist=1, bear=270)
    if ens_num==29:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=0)
        return compute_displacement(newlat, newlon, dist=2, bear=270)
    if ens_num==30:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=0)
        return compute_displacement(newlat, newlon, dist=3, bear=270)
    if ens_num==31:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=180)
        return compute_displacement(newlat, newlon, dist=1, bear=90)
    if ens_num==32:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=180)
        return compute_displacement(newlat, newlon, dist=2, bear=90)
    if ens_num==33:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=180)
        return compute_displacement(newlat, newlon, dist=3, bear=90)
    if ens_num==34:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=180)
        return compute_displacement(newlat, newlon, dist=1, bear=90)
    if ens_num==35:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=180)
        return compute_displacement(newlat, newlon, dist=2, bear=90)
    if ens_num==36:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=180)
        return compute_displacement(newlat, newlon, dist=3, bear=90)
    if ens_num==37:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=180)
        return compute_displacement(newlat, newlon, dist=1, bear=90)
    if ens_num==38:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=180)
        return compute_displacement(newlat, newlon, dist=2, bear=90)
    if ens_num==39:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=180)
        return compute_displacement(newlat, newlon, dist=3, bear=90)
    if ens_num==40:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=180)
        return compute_displacement(newlat, newlon, dist=1, bear=270)
    if ens_num==41:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=180)
        return compute_displacement(newlat, newlon, dist=2, bear=270)
    if ens_num==42:
        newlat, newlon=compute_displacement(lat, lon, dist=1, bear=180)
        return compute_displacement(newlat, newlon, dist=3, bear=270)
    if ens_num==43:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=180)
        return compute_displacement(newlat, newlon, dist=1, bear=270)
    if ens_num==44:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=180)
        return compute_displacement(newlat, newlon, dist=2, bear=270)
    if ens_num==45:
        newlat, newlon=compute_displacement(lat, lon, dist=2, bear=180)
        return compute_displacement(newlat, newlon, dist=3, bear=270)
    if ens_num==46:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=180)
        return compute_displacement(newlat, newlon, dist=1, bear=270)
    if ens_num==47:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=180)
        return compute_displacement(newlat, newlon, dist=2, bear=270)
    if ens_num==48:
        newlat, newlon=compute_displacement(lat, lon, dist=3, bear=180)
        return compute_displacement(newlat, newlon, dist=3, bear=270)

############################################################################
############################################################################

def compute_displacement(lat, lon, dist=None, bear=None):
    
    """Compute the latitude and longitude for the respective ensemble member.
    
    """
    if not dist:
        return lat, lon
    if dist:
        R = 6378.1 #Radius of the Earth (km)
        brng = math.radians(bear) #Bearing is 90 degrees converted to radians.
        d = dist #Distance in km
        lat1 = math.radians(lat) #Current lat point converted to radians
        lon1 = math.radians(lon) #Current long point converted to radians
        lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
               math.cos(lat1)*math.sin(d/R)*math.cos(brng))
        lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),
               math.cos(d/R)-math.sin(lat1)*math.sin(lat2))
        lat2 = math.degrees(lat2)
        lon2 = math.degrees(lon2)
        return lat2, lon2

############################################################################
############################################################################

def height_generator(ens_num, altitude):
    
    """Generate the height for the respective ensemble member.
    
    """
    fraction=np.hstack([np.array([1 for i in range(49)]),
                        np.array([0.95 for i in range(49)]),
                        np.array([0.9 for i in range(49)]),
                        np.array([0.85 for i in range(49)]),
                        np.array([0.8 for i in range(49)]),
                        np.array([0.75 for i in range(49)]),
                        np.array([0.7 for i in range(49)])
                       ])
    return altitude*fraction[ens_num]

############################################################################
############################################################################

#where is hysplit working folder?
working_dir = f'/glade/scratch/molina/hysplit/trunk/{which_working}' 
#where is arl format meteo data?
meteo_dir = f'/glade/scratch/molina/basile/{which_climate}'
#where is hysplit model executable?
hysplit_dir=r'/glade/scratch/molina/hysplit/trunk/exec/hyts_std'
#where to put trajs?
output_dir=f'/glade/scratch/molina/basile/{which_climate}_traj/'

############################################################################
############################################################################

runtime = -240

basename = []
years = []
months = []
hours = []
location = []
altitudes = []

for i in range(len(csv_file)):

    print(i)
    basename = 'trajid'+str(csv_file.iloc[i][0])+'_subregion'+str(csv_file.iloc[i][2])+'_'+'ens'+str(ens_number)+'_'

    years = [csv_file.iloc[i][8]]        
    months = [csv_file.iloc[i][9]]
    hours = [csv_file.iloc[i][11]]
    location = ens_create(ens_num=ens_number, lat=csv_file.iloc[i][4], lon=csv_file.iloc[i][5])
    altitudes = [height_generator(ens_num=ens_number, altitude=csv_file.iloc[i][6])]
    day1 = (csv_file.iloc[i][10]-1)
    day2 = csv_file.iloc[i][10]

    generate_bulktraj(basename=basename, 
                      hysplit_working=working_dir, 
                      output_dir=output_dir, 
                      meteo_dir=meteo_dir, 
                      years=years, 
                      months=months, 
                      hours=hours, 
                      altitudes=altitudes, 
                      coordinates=location, 
                      run=runtime,
                      meteoyr_2digits=False, outputyr_2digits=False,         
                      monthslice=slice(day1, day2, 1),
                      meteo_bookends=([1] , [1]),
                      get_reverse=False, get_clipped=False, hysplit=hysplit_dir)


############################################################################
############################################################################
############################################################################ 
