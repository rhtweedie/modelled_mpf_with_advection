import xarray as xr
import h5py as h5
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from os import listdir
from scipy.interpolate import griddata


def main():
    # find years which are available for both MPF and SIR
    years_lee = listdir(f'/home/sl/melt_pond/data')
    years_lee.remove('mp_coordinate.nc')
    years_lee.remove('2001')
    years_lee.remove('2002')
    years_lee.remove('2003')

    fns_misr = listdir(f'/home/ssureen/MISR_data_monthly')
    years_misr = []
    for fn in fns_misr:
        years_misr.append(fn[6:10])

    years = list(set(years_lee) & set(years_misr))
    print(f'Available years: {years}')

    # set months to retrieve and plot
    months = ['06', '07', '08']

    R0 = 65.43
    L = 16.14
    TAU = 5.15
    HNET = 0.025    # hnet between 20 and 40mm from Landy

    # retrieve MPF coordinates and convert to EASE grid for comparison
    try:
        lon_MPF, lat_MPF = MPF_UCL_COORDS('/home/sl/melt_pond/data/mp_coordinate.nc')
        x_MPF, y_MPF = WGS84toEASE2N(lon_MPF, lat_MPF)
        print(f'MPF coordinates successfully retrieved.')
    except Exception as e:
        print(f'MPF coordinates could not be retrieved: {e}')
        return

    annual_r2 = []

    for year in years:

        print(f"======== Processing {year} ========")

        # retrieve MISR data for given year
        try:
            MISR_path = f'/home/ssureen/MISR_data_monthly/April {year} Roughness.h5'
            MISR, lon_MISR, lat_MISR, x_MISR, y_MISR = load_MISR(MISR_path)
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
            mpf_interpolated = interpolate_to_MISR(x_MPF, y_MPF, mpf_mean ,x_MISR, y_MISR )
            masked_mpf, masked_mean_MISR, masked_lat_mean_MISR = MASK_MPF_MISR(mpf_interpolated, MISR, lat_MISR)
            print('Interpolated Mean')
        except Exception as e:
            print(f'Could not interpolate data: {e}')
            return

        # predict MPF using constants provided by Landy et al, 2015
        pred_mpf = predict_mpf(MISR, R0, L, TAU, HNET)

        # to find the overlap - gives only the points that exist in both sets
        mpf_overlap = masked_mpf + (0 * pred_mpf)
        pred_overlap = pred_mpf + (0 * masked_mpf)

        fig = plt.figure(figsize=(10,10))
        plt.set_cmap('bwr')

        plt.subplot(221)
        plt.pcolormesh(mpf_overlap[1000:7000:10, 1000:7000:10], vmin=0, vmax=0.6)
        plt.colorbar()
        plt.title(f'Observed MPF Mean {year}')

        plt.subplot(222)
        plt.pcolormesh(pred_overlap[1000:7000:10, 1000:7000:10], vmin=0, vmax=0.6)
        plt.colorbar()
        plt.title(f'Predicted MPF Mean {year}')

        plt.subplot(223)
        plt.pcolormesh(pred_overlap[1000:7000:10, 1000:7000:10] - mpf_overlap[1000:7000:10, 1000:7000:10], vmin=-0.4, vmax=0.4)
        plt.colorbar()
        plt.title('Predicted - Observed')

        valid_indices = ~np.isnan(mpf_overlap) & ~np.isnan(pred_overlap)
        mpf_overlap_sub = mpf_overlap[valid_indices]
        pred_overlap_sub = pred_overlap[valid_indices]

        # calculate r^2 (Pearson's product) between true and predicted MPF
        r2 = np.corrcoef(mpf_overlap_sub, pred_overlap_sub)[0,1]
        annual_r2.append(r2)
        print(f'r^2: {r2}')

        plt.subplot(224)
        plt.hist2d(mpf_overlap_sub, pred_overlap_sub, bins=(50,50), cmap='Blues')
        plt.xlabel("Observed MPF")
        plt.ylabel("Predicted MPF")
        plt.title("Predicted vs Observed")
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.axline((0, 0), slope=1)
        plt.text(0.1, 0.9, r2)

        try:
            fig_fn = f'/home/htweedie/melt_ponds/Lee_MPF_figs/Lee_pred/MPF_pred_{year}.png'
            plt.savefig(fig_fn, format='png')
            print(f"{year} figure saved at {fig_fn}.")
        except Exception as e:
            print(f"Figure could not be saved: {e}")



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
    Interpolates data of the shape x_in, y_in to the shape of x_out, y_out.
    
    Params:
        data: the data to be interpolated
        x_in, y_in: the shape of the data to be interpolated
        
    Returns:
        x_out, y_out: the shape to which the data will be interpolated'''
    return griddata((x_in.ravel(), y_in.ravel()), data.ravel( ), (x_out.ravel(), y_out.ravel()), 'nearest').reshape(8000,8000) 


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


def predict_mpf(SIR, R0, l, tau, hnet):
    '''
    Predicts meltpond fraction given an input SIR, based on the model by Landy et al, 2015.
    '''
    R = R0 * np.exp(-l * SIR) + tau
    return (1 - np.exp(-R * hnet))



if __name__ == "__main__":
    main()
