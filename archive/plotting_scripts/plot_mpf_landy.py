# Importing modules to access and visualise data
import xarray as xr # used for netcdf and h5 files, climate data
import h5py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfea
import pyproj
import datetime
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import os, sys
from matplotlib.colors import LinearSegmentedColormap


def main():
    years = os.listdir(f'/home/sl/melt_pond/data')
    years.remove('mp_coordinate.nc')
    years.remove('2001')
    years.remove('2002')
    years.remove('2003')

    # retrieve MPF coordinates and convert to EASE grid for comparison
    try:
        lon_MPF, lat_MPF = MPF_UCL_COORDS('/home/sl/melt_pond/data/mp_coordinate.nc')
        x_MPF, y_MPF = WGS84toEASE2N(lon_MPF, lat_MPF)
        print(f'MPF coordinates successfully retrieved.')
    except Exception as e:
        print(f'MPF coordinates could not be retrieved: {e}')
        return

    for year in years:
        print(f'======== Processing {year} ========')
        # retrieve MISR data for given year
        try:
            MISR_path = f'/home/ssureen/MISR_data_monthly/April {year} Roughness.h5'
            MISR, lon_MISR, lat_MISR, x_MISR, y_MISR = LOAD_MISR(MISR_path)
            print(f'Succesfully retrieved MISR data from {MISR_path}.')
        except Exception as e:
            print(f'MISR data could not be retrieved from {MISR_path}: {e}')
            return
        
        # retrieve MPF data for given year
        try:
            mpf_path = f'/home/sl/melt_pond/data/{year}/MODIS_Meltpond_Fraction_CPOM_5km_monthly_{year}.nc'
            mpf_mean, mpf_may, mpf_june, mpf_july, mpf_aug = LOAD_MPF(mpf_path)
            print(f'Successfully retrieved MPF data from {mpf_path}.')
        except Exception as e:
            print(f'MPF data could not be retrieved from {mpf_path}: {e}')
            return

        try:
            print('Interpolating...')
            # interpolate MPF to MISR grid for each month and mean for the year
            mpf_mean_MISRGRID = interpolate_to_MISR(x_MPF, y_MPF, mpf_mean ,x_MISR, y_MISR )
            masked_MPF_mean_MISRGRID, masked_mean_MISR, masked_lat_mean_MISR = MASK_MPF_MISR(mpf_mean_MISRGRID, MISR, lat_MISR)
            print('Interpolated Mean')

            mpf_may_MISRGRID = interpolate_to_MISR(x_MPF, y_MPF, mpf_may ,x_MISR, y_MISR )
            masked_MPF_may_MISRGRID, masked_may_MISR, masked_lat_may_MISR = MASK_MPF_MISR(mpf_may_MISRGRID, MISR, lat_MISR)
            print('Interpolated May')

            mpf_june_MISRGRID = interpolate_to_MISR(x_MPF, y_MPF, mpf_june ,x_MISR, y_MISR )
            masked_MPF_june_MISRGRID, masked_june_MISR, masked_lat_june_MISR = MASK_MPF_MISR(mpf_june_MISRGRID, MISR, lat_MISR)
            print('Interpolated June')

            mpf_july_MISRGRID = interpolate_to_MISR(x_MPF, y_MPF, mpf_july ,x_MISR, y_MISR )
            masked_MPF_july_MISRGRID, masked_july_MISR, masked_lat_july_MISR = MASK_MPF_MISR(mpf_july_MISRGRID, MISR, lat_MISR)
            print('Interpolated July')

            mpf_aug_MISRGRID = interpolate_to_MISR(x_MPF, y_MPF, mpf_aug ,x_MISR, y_MISR )
            masked_MPF_aug_MISRGRID, masked_aug_MISR, masked_lat_aug_MISR = MASK_MPF_MISR(mpf_aug_MISRGRID, MISR, lat_MISR)
            print('Interpolated August')

            print(f'Succesfully interpolated all months.')
        except Exception as e:
            print(f'Could not interpolate data: {e}')
            return

        fig = plt.figure(figsize=(10,10))

        plt.subplot(221)
        plt.pcolormesh(masked_MPF_may_MISRGRID[2500:6000:10, 2000:6000:10])
        plt.title('Observed MPF May 2020')

        plt.subplot(222)
        plt.pcolormesh(masked_MPF_june_MISRGRID[2500:6000:10, 2000:6000:10])
        plt.title('Observed MPF June 2020')

        plt.subplot(223)
        plt.pcolormesh(masked_MPF_july_MISRGRID[2500:6000:10, 2000:6000:10])
        plt.title('Observed MPF July 2020')

        plt.subplot(224)
        plt.pcolormesh(masked_MPF_mean_MISRGRID[2500:6000:10, 2000:6000:10])
        plt.title('Observed MPF Mean 2020')

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes((0.85, 0.1, 0.075, 0.8))
        plt.colorbar(cax=cax)

        try:
            fig_fn = f'/home/htweedie/melt_ponds/MPF_figs/MPF_{year}.png'
            plt.savefig(fig_fn, format='png')
            print(f'Figure saved as {fig_fn}.')
        except Exception as e:
            print(f'{year} figure could not be saved: {e}')
            return


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


def LOAD_MISR(MISR_path):
    '''Loads MISR data and coordinates from specified file path.

    Params:
        MISR_path (str): the file path from which to retrieve data

    Returns:
        data (np.array): roughness data retrieved from the specified file
        lon, lat
        x, y
    '''

    file = h5py.File(MISR_path, 'r')
    
    # extract coord data
    lon = np.array(file['GeoLocation']['Longitude'])
    lat = np.array(file['GeoLocation']['Latitude'])
    x = np.array(file['GeoLocation']['x'])
    y = np.array(file['GeoLocation']['y'])

    # extract roughness data
    data = np.array(file['Roughness']['Roughness_2D_svm'])    
    
    file.close()

    return data, lon, lat, x, y


def LOAD_MPF(MPF_path): # Loads data from Sangyun Lee dataset on CPOM servers, just for month of JULY
    '''Loads MPF data for individual months and calculates the pixel-by-pixel mean across all months.

    Params:
        MPF_path (str): the file path from which data will be retrieved

    Returns:
        mean, may, june, july, aug: numpy arrays for the mean and individual months
    '''
    ds = xr.open_dataset(MPF_path)

    # retrieve data for individual months
    may = np.array(ds['may_monthly'])
    june = np.array(ds['june_monthly'])
    july = np.array(ds['july_monthly'])
    aug = np.array(ds['august_monthly'])
 
    # calculate mean over all months
    mean = np.nanmean(np.array([may, june, july, aug]), 0)

    return mean, may, june, july, aug


def MPF_UCL_COORDS(MPF_coords_path): # reads coordinate data from sangyun lee mpf data set
    '''
    Retrieves grid coordinates for MPF data.
    
    Params:
        MPF_coords_path (str): the file path from which the coordinates will be retrieved
        
    Returns:
        lon, lat
    '''
    ds = xr.open_dataset(MPF_coords_path)
    lon = np.asarray(ds['mp_lon'])
    lat = np.asarray(ds['mp_lat'])
        
    return lon, lat


def MASK_MPF_MISR(mpf_MISRGRID, MISR, lat_MISR):
    # Create masks for valid data in each array
    mask1 = ~np.isnan(mpf_MISRGRID)  # Invert the NaN values to get a mask of valid data
    mask2 = ~np.isnan(MISR)

    # Create a joint mask where both arrays have valid data
    joint_mask = mask1 & mask2

    # Use the joint mask to apply the mask to both arrays and corresponding latitude
    masked_MPF_MISRGRID = np.ma.masked_array(mpf_MISRGRID, mask=~joint_mask)
    masked_MISR = np.ma.masked_array(MISR, mask=~joint_mask)
    masked_lat_MISR = np.ma.masked_array(lat_MISR, mask=~joint_mask)

    return masked_MPF_MISRGRID, masked_MISR, masked_lat_MISR


def interpolate_to_MISR(x_in, y_in, data, x_out, y_out):
    '''
    Interpolates data of the shape X_in, Y_in to the shape of X_out, Y_out.
    
    Params:
        data: the data to be interpolated
        x_in, y_in: the shape of the data to be interpolated
        
    Returns:
        x_out, y_out: the shape to which the data will be interpolated'''
    return griddata((x_in.ravel(), y_in.ravel()), data.ravel( ), (x_out.ravel(), y_out.ravel()), 'nearest').reshape(8000,8000) 


if __name__ == '__main__':
    main()
