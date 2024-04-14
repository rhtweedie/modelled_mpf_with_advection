import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset, MFDataset
import xarray as xr
import pandas as pd
from scipy import spatial
from datetime import datetime
from datetime import timedelta
import os.path
import h5py as h5
from scipy.spatial import KDTree
import pyproj
from scipy.interpolate import griddata

def main():

    # 2010 is the first complete year of ice drift data. 
    years = [2010, 2011, 2017, 2018, 2019, 2020]

    for year in years:
        print(f'------ Processing {year} ------')

        YEAR = year
        START_DATE = f'{YEAR}-04-01 12:00:00'
        SPACING = 8
        DAYS_TO_FORWARD = 183
        delta_t = 86400  # in seconds

        # retrieve MPF coordinates
        coord_fn = '/home/htweedie/melt_ponds/data/OLCI/olci/LongitudeLatitudeGrid-n12500-Arctic.h5'
        coords = h5.File(coord_fn, 'r')
        mpf_lon =  np.array(coords['Longitudes'])
        mpf_lat = np.array(coords['Latitudes'])
        x_mpf, y_mpf = WGS84toEASE2N(mpf_lon, mpf_lat)

        # retrieve MISR data and coordinates
        fn = f'/home/ssureen/MISR_data_monthly/April {YEAR} Roughness.h5'
        sigma, sigma_lon, sigma_lat, sigma_x, sigma_y = load_MISR(fn)

        # take an even subset of the data to reduce computational requirements
        all_lats = sigma_lat[::SPACING, ::SPACING].ravel()
        all_lons = sigma_lon[::SPACING, ::SPACING].ravel()
        all_sigma = sigma[::SPACING, ::SPACING].ravel()

        num_points = len(all_lons)
        earliest_date = START_DATE
        advect_start_date = START_DATE
        dates = [advect_start_date]

        # create dataframes in which lat, lon and mpf data will be stored
        lons = np.zeros((1, num_points))*np.nan
        lats = np.zeros((1, num_points))*np.nan
        mpfs = np.zeros((1, num_points))*np.nan
        lats_df = pd.DataFrame(data=lats, index=dates)
        lons_df = pd.DataFrame(data=lons, index=dates)
        mpfs_df = pd.DataFrame(data=mpfs, index=dates)

        lats_df.loc[advect_start_date] = all_lats # np.arange(90.,110.,1.)
        lons_df.loc[advect_start_date] = all_lons # np.arange(90.,110.,1.)

        # format the starting date, then load MPF data for that day
        date = datetime.strptime(advect_start_date, "%Y-%m-%d %H:%M:%S")
        start_YYYYMMDD = date.strftime("%Y")+date.strftime("%m")+date.strftime("%d")
        try:
            if YEAR >= 2017 and YEAR <= 2023:
                fn = f'/home/htweedie/melt_ponds/data/OLCI/olci/{YEAR}/data/mpd1_{start_YYYYMMDD}.nc'
            elif YEAR >= 2002 and YEAR <= 2011:
                fn = f'/home/htweedie/melt_ponds/data/MERIS/mecosi/{YEAR}/data/mpd1_{start_YYYYMMDD}.nc'
            ds = xr.open_dataset(fn)
            mpf = ds['mpf']
        except Exception as e:
            mpf = np.zeros(num_points)
            mpf[:] = np.nan
            print(f'Data could not be retrieved for advection start date, {date}: {e}')

        # find all MPFs within each sigma grid cell
        tree = KDTree(list(zip(x_mpf.ravel(), y_mpf.ravel())))
        x_sigma, y_sigma = WGS84toEASE2N(all_lons, all_lats)
        max_radius = 10000
        indices_within_grid = tree.query_ball_point(list(zip(x_sigma, y_sigma)), r = max_radius)

        # calculate the mean MPF within the radius for each sigma grid cell, and add to df
        if len(indices_within_grid) > 0:
            mean_mpf = np.zeros(num_points)
            for i in range(num_points):
                mean_mpf[i] = np.mean(np.asarray(mpf).ravel()[indices_within_grid[i]])
        mean_mpf[mean_mpf==0] = np.nan
        mpfs_df.loc[advect_start_date] = np.asarray(mean_mpf).ravel()

        # initialise 'Bouys' for all points to be advected
        points = Buoys(lons_df.loc[advect_start_date], lats_df.loc[advect_start_date], advect_start_date, earliest_date)

        # advect all points by the set number of days
        forwarded_mpfs = []
        for i in np.arange(1, DAYS_TO_FORWARD+1):
            print(f'This is year {YEAR}, loop #{i}')

            Ufield, Vfield, lon_start, lat_start = loaddate_ofOSISAF(points.getdate(), hemisphere='nh')
            U,V = find_UV_atbuoy_pos(lon_start, lat_start, Ufield.flatten(),Vfield.flatten(), points)

            # don't advect buoys when there is no ice
            fixed=np.logical_or(U.mask, V.mask)
            U[fixed]=0.
            V[fixed]=0.

            LON,LAT = points.trajectory(U, V, delta_t=delta_t) # U,V in m/s, delta_t in seconds

            # create dataframe with new lats and lons
            new_lons = pd.DataFrame(LON.rename(points.getdate())).T
            new_lats = pd.DataFrame(LAT.rename(points.getdate())).T

            # add dataframe with new lats and lons to original one
            lons_df = pd.concat([lons_df, new_lons])
            lats_df = pd.concat([lats_df, new_lats])
            
            x_sigma, y_sigma = WGS84toEASE2N(new_lons, new_lats)

            # get and format current datestring
            date = datetime.strptime(points.getdate(), "%Y-%m-%d %H:%M:%S")
            YYYYMMDD = date.strftime("%Y")+date.strftime("%m")+date.strftime("%d")

            # retrieve MPF data for this day
            try:
                if YEAR >= 2017 and YEAR <= 2023:
                    fn = f'/home/htweedie/melt_ponds/data/OLCI/olci/{YEAR}/data/mpd1_{YYYYMMDD}.nc'
                elif YEAR >= 2002 and YEAR <= 2011:
                    fn = f'/home/htweedie/melt_ponds/data/MERIS/mecosi/{YEAR}/data/mpd1_{YYYYMMDD}.nc'
                ds = xr.open_dataset(fn)
                mpf = ds['mpf']
                print(f'Data retrieved for {date}')
            except Exception as e:
                mpf = np.zeros(896*608)
                mpf[:] = np.nan
                print(f'Data could not be retrieved for {date}: {e}')

            # Query the tree to find all points within final_lons and final_lats grids
            max_radius = 10000
            indices_within_grid = tree.query_ball_point(list(zip(x_sigma.ravel(), y_sigma.ravel())), r = max_radius)

            # calculate the mean MPF within the radius for each sigma grid point
            if len(indices_within_grid) > 0:
                mean_mpf = np.zeros(num_points)
                for i in range(num_points):
                    mean_mpf[i] = np.mean(np.asarray(mpf).ravel()[indices_within_grid[i]])

            # convert 0s to nans and append to list
            mean_mpf[mean_mpf==0] = np.nan
            forwarded_mpfs.append(mean_mpf)

            ind = format_date(date.strftime("%Y"), date.strftime("%m"), date.strftime("%d"))
            new_mpfs = pd.DataFrame(data=np.asarray(mean_mpf).reshape(1, len(np.asarray(mean_mpf).ravel())), index=[ind])
            mpfs_df = pd.concat([mpfs_df, new_mpfs])

        # save dataframe
        mpfs_df.to_pickle(f'/home/htweedie/melt_ponds/data/forwarded_mpfs/mpf_from_{start_YYYYMMDD}_{DAYS_TO_FORWARD} days_spacing_{SPACING}.pkl')
        lons_df.to_pickle(f'/home/htweedie/melt_ponds/data/forwarded_mpfs/lon_from_{start_YYYYMMDD}_{DAYS_TO_FORWARD} days_spacing_{SPACING}.pkl')
        lats_df.to_pickle(f'/home/htweedie/melt_ponds/data/forwarded_mpfs/lat_from_{start_YYYYMMDD}_{DAYS_TO_FORWARD} days_spacing_{SPACING}.pkl')
        print(f'Dataframes saved.')



class Buoys:
    
    global rad, r_earth
    rad=np.pi/180.0 # radiant <-> degree
    r_earth=6.3675*10**6 # radius of Earth in [m]
    
    def __init__(self, lon_start, lat_start, earliest_date_of_buoy, start_advect_date):
        print(lon_start)
        self.oldlon = lon_start * rad
        self.oldlat = lat_start * rad
        self.lon = lon_start * rad
        self.lat = lat_start * rad
        self.initlon = lon_start * rad
        self.initlat = lat_start * rad
        self.old_u = np.zeros(lon_start.shape)
        self.old_v = np.zeros(lon_start.shape)
        self.date = datetime.strptime(earliest_date_of_buoy, "%Y-%m-%d %H:%M:%S")
        self.startdates = start_advect_date
        #self.delta_x = np.zeros(lon_start.shape)
        #self.delta_y = np.zeros(lon_start.shape)
        #self.u_ice = np.zeros(lon_start.shape)
        #self.v_ice = np.zeros(lon_start.shape)
        
    def getdate(self):
        return self.date.strftime("%Y-%m-%d %H:%M:%S")
        
    def trajectory(self, new_u, new_v, delta_t):
        #print("Update buoy positions. Integrate for " + str(delta_t/3600.) + " hours.")
        
        #save old position in case the drifter leaves the domain
        self.oldlon = self.lon # radiant
        self.oldlat = self.lat # radiant
        
        #displacement vectors
        deltax1 = self.old_u * delta_t
        deltay1 = self.old_v * delta_t
        deltax2 = new_u * delta_t
        deltay2 = new_v * delta_t
        
        #Heun method (2nd order)
        self.lon = self.lon + (0.5*(deltax1 + deltax2) / (r_earth*np.cos(self.lat.values)) )
        self.lat = self.lat + (0.5*(deltay1 + deltay2) /  r_earth )
        
        # keep degree in range 0..360 and -90..90
        lon_deg = self.lon/rad % 360
        lat_deg = np.clip(self.lat/rad, -90., 90.)
        self.lon = lon_deg*rad
        self.lat = lat_deg*rad
        
        #update velocity here (old value was needed for heun method)
        self.old_u=new_u
        self.old_v=new_v
        
        # set positions to NaN before the buoy is supposed to move
        #idx=getindices_beforestart(self.getdate(), self.startdates)
        #lon_deg[idx] = np.nan
        #lat_deg[idx] = np.nan
        #self.lon[idx] = self.initlon[idx]
        #self.lat[idx] = self.initlat[idx]
        #self.old_u[idx]=0.
        #self.old_v[idx]=0.
        
        # update time stamp
        self.date = self.date + timedelta(seconds=delta_t)

        return lon_deg, lat_deg



# a useful function we'll need
def length_of_latitude_circle(lat=85.):
    r_earth=6.3675*10**6 # radius of Earth in [m]
    rad=np.pi/180.0 # radiant <-> degree  
    return 2*np.pi*r_earth*np.cos(lat*rad) / 1000. # km


# find the buoys that are not to be advected yet (current date < start date)
def getindices_beforestart(currentdate, startdates):
        
    indices=np.zeros(np.shape(startdates),dtype='bool')
    for i,val in enumerate(startdates):
        # don't advect yet
        if currentdate < startdates[i]:
            indices[i]=True
            
    return indices


# load OSISAF data for Northern Hemisphere at a certain date
def loaddate_ofOSISAF(datestring, hemisphere='nh'):
    
    # convert datestring to datetime object
    thedate = datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S")
    
    # let's construct the file name, 
    # e.g. drift-velocities/archive/ice/drift_lr/merged/2019/09/
    # ice_drift_nh_polstere-625_multi-oi_201909011200-201909031200.nc
    pathtofile = "/home/htweedie/melt_ponds/data/drift-velocities/archive/ice/drift_lr/merged/"
    # middle part
    middlefilename="ice_drift_"+hemisphere+"_polstere-625_multi-oi_"
    # e.g. 201907291200-201907311200 (48hr span)
    enddate=thedate + timedelta(days=2)
    # YYYY/MM/ (from end date)
    YYYYMM=enddate.strftime("%Y")+"/"+enddate.strftime("%m")+"/"
    endfilename= thedate.strftime("%Y%m%d%H%M") + "-" + enddate.strftime("%Y%m%d%H%M") + '.nc'
    
    # the OSISAF file to be loaded
    filename=pathtofile + YYYYMM + middlefilename + endfilename
    
    # take previous files in case there is a data gap
    sd=thedate
    ed=enddate
    while os.path.isfile(filename)!=True:
        # try previous file
        sd=sd - timedelta(days=1)
        ed=ed - timedelta(days=1)
        # YYYY/MM/ (from end date)
        YYYYMM=ed.strftime("%Y")+"/"+ed.strftime("%m")+"/"
        endfilename= sd.strftime("%Y%m%d%H%M") + "-" + ed.strftime("%Y%m%d%H%M") + '.nc'
        filename=pathtofile + YYYYMM + middlefilename + endfilename
        print('data gap: try previous file '+filename+' ...')
    
    #print("loading "+filename+ " ...") # Python3 needs brackets here
    
    # load the file
    fl = Dataset(filename)
    #xc=fl.variables['xc']
    #yc=fl.variables['yc']
    #XC,YC=np.meshgrid(xc,yc)
    
    # lon lat on grid
    lon_start=np.copy(fl.variables['lon'])
    lat_start=np.copy(fl.variables['lat'])

    # lon lat at the end of the displacement
    lon_end=np.squeeze(fl.variables['lon1'][0,:,:])
    lat_end=np.squeeze(fl.variables['lat1'][0,:,:])
    
    # close the file
    fl.close()
    
    # compute Ufield from end points and start points (48hour change)
    deltalon=lon_end-lon_start
    deltalon[deltalon>100.]=deltalon[deltalon>100.]-360.   # jump at -180..180
    deltalon[deltalon<-100.]=deltalon[deltalon<-100.]+360. # jump at -180..180
    Ufield=deltalon/48. *length_of_latitude_circle(lat=lat_start[:,:])/360. / 3.6 # km/h -> m/s
    
    # compute Vfield as well
    Vfield=(lat_end-lat_start)/48. *length_of_latitude_circle(lat=0.)/360. / 3.6 #km/h -> m/s
    
    return Ufield, Vfield, lon_start, lat_start


# nearest-neighbor interpolation, finds U,V at the position of the buoys using a fast KDTree method
def find_UV_atbuoy_pos(lon_start,lat_start, Ufield,Vfield, objects):
    
    # (lon,lat) tuples of the underlying grid
    A = np.array([lon_start[:,:].flatten(), lat_start[:,:].flatten()]).T # -180..180 assumed in OSISAF
    # change to -180..180 as assumed in OSISAF data; in the trajectory code its 0..360
    lon_adjust = objects.lon/rad
    lon_adjust[lon_adjust>180.] = lon_adjust[lon_adjust>180.]-360.
    # zip buoy (lon & lat) arrays to (lon,lat) tuples
    tuples = np.column_stack((lon_adjust, objects.lat/rad)) 
    # fast KDTree nearest neighbor method
    idx = spatial.KDTree(A).query(tuples)[1]
    
    return Ufield[idx], Vfield[idx]


def WGS84toEASE2N(lon, lat):
    '''Converts WGS84 coordinates to EASE2N.

    Params:
        lon (array): the WGS84 longitude to convert
        lat (array): the WGS84 latitude to convert

    Returns:
        (x, y): the corresponding EASE2N x and y coordinates
    '''

    proj_EASE2N = pyproj.Proj("+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
    proj_WGS84 = pyproj.Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ")
    return pyproj.transform(proj_WGS84, proj_EASE2N, lon, lat)


def EASE2NtoWGS84(x, y):

    EASE2 = "+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = pyproj.Transformer.from_crs(EASE2, WGS84)
    lon, lat = transformer.transform(x, y)
    return lon, lat


def load_MISR(MISR_path):
    '''Loads MISR data and coordinates from specified file path.

    Params:
        MISR_path (str): the file path from which to retrieve data

    Returns:
        data (np.array): roughness data retrieved from the specified file
        lon, lat
        x, y
    '''

    file = h5.File(MISR_path, 'r')
    
    # extract coord data
    lon = np.array(file['GeoLocation']['Longitude'])
    lat = np.array(file['GeoLocation']['Latitude'])
    x = np.array(file['GeoLocation']['x'])
    y = np.array(file['GeoLocation']['y'])

    # extract roughness data
    data = np.array(file['Roughness']['Roughness_2D_svm'])    
    
    file.close()

    return data, lon, lat, x, y


def predict_mpf(SIR, R0, l, tau, hnet):
    '''
    Predicts meltpond fraction given an input SIR, based on the model by Landy et al, 2015.
    '''
    R = R0 * np.exp(-l * SIR) + tau
    return (1 - np.exp(-R * hnet))


def interpolate_to_MISR(x_in, y_in, data, x_out, y_out):
    '''
    Interpolates data of the shape x_in, y_in to the shape of x_out, y_out.
    
    Params:
        data: the data to be interpolated
        x_in, y_in: the shape of the data to be interpolated
        
    Returns:
        x_out, y_out: the shape to which the data will be interpolated'''
    return griddata((x_in.ravel(), y_in.ravel()), data.ravel( ), (x_out.ravel(), y_out.ravel()), 'nearest').reshape(8000,8000) 


def format_date(year, month, day):
    return f"{year}-{month}-{day} 12:00:00"



if __name__ == "__main__":
    main()
