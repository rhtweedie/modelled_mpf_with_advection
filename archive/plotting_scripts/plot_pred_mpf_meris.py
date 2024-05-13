import xarray as xr
import h5py as h5
import numpy as np
import pyproj
import matplotlib.pyplot as plt
from os import listdir
from scipy.interpolate import griddata


def main():
    # find years which are available for both MPF and SIR
    years_meris = listdir(f'/home/htweedie/melt_ponds/data/MERIS/mecosi')
    fns_misr = listdir(f'/home/ssureen/MISR_data_monthly')
    years_misr = []
    for fn in fns_misr:
        years_misr.append(fn[6:10])

    years = list(set(years_meris) & set(years_misr))
    print(f'Available years: {years}')

    # set months to retrieve and plot
    months = ['06', '07', '08']

    R0 = 65.43
    L = 16.14
    TAU = 5.15
    HNET = 0.025    # hnet between 20 and 40mm from Landy

    for year in years:

        print(f"======== Processing {year} ========")

        annual_r2 = []
        monthly_mpfs = []

        print(f"Retrieving MPF data and calculating monthly means:")
        for month in months:
            # set the number of days for the given month
            match month:
                case '05' | '07' | '08':
                    days = 31
                case '06':
                    days = 30

            # initialise empty list to store mpf for each day
            daily_mpfs = []
            for day in range(days):
                # start days from 1 rather than 0
                day += 1
                # configure days as strings to work with file names
                if day < 10:
                    day = f'0{day}'
                else:
                    day = str(day)

                # retrieve data for given day. If the file doesn't exist, skip it.
                try:
                    fn = f'/home/htweedie/melt_ponds/data/MERIS/mecosi/{year}/data/mpd1_{year}{month}{day}.nc'
                    ds = xr.open_dataset(fn)
                    mpf = ds['mpf']
                    daily_mpfs.append(np.array(mpf))
                except Exception as e:
                    print(f'Data could not be retrieved for {day}/{month}/{year}: {e}')
                    continue

            # calculate the mean of the current month and save as .npy
            month_mean = np.nanmean(daily_mpfs, 0)
            monthly_mpfs.append(month_mean)
            fn = f'/home/htweedie/melt_ponds/data/MERIS/mecosi/{year}/mean_{month}_{year}.npy'
            np.save(fn, month_mean)
            print(f"{month} mean calculated and saved at {fn}.")

        # find mean of all months
        mean_summer_mpf = np.nanmean(monthly_mpfs, 0)
        print(f"{year} mean calculated.")
    
        # retrieve MPF coordinates
        try:
            coord_fn = '/home/htweedie/melt_ponds/data/MERIS/LongitudeLatitudeGrid-n12500-Arctic.h5'
            coords = h5.File(coord_fn, 'r')
            mpf_lon =  np.array(coords['Longitudes'])
            mpf_lat = np.array(coords['Latitudes'])
            x_mpf, y_mpf = WGS84toEASE2N(mpf_lon, mpf_lat)
            print("MPF coordinates retrieved.")
        except Exception as e:
            print(f"MPF coordinates could not be retrieved: {e}")

        # retrieve MISR data and coordinates
        try:
            MISR_path = f'/home/ssureen/MISR_data_monthly/April {year} Roughness.h5'
            MISR, lon_MISR, lat_MISR, x_MISR, y_MISR = load_MISR(MISR_path)
            print(f"MISR data retrieved for {year}")
        except Exception as e:
            print(f"MISR data could not be retrieved for {year}: {e}")

        # interpolate MPF data to MISR grid coordinates and save as .npy
        try:   
            print("Interpolating...")
            mpf_interpolated = interpolate_to_MISR(x_mpf, y_mpf, mean_summer_mpf, x_MISR, y_MISR)
            fn = f'/home/htweedie/melt_ponds/data/interpolated_mpf/interpolated_mpf_{year}'
            np.save(fn, mpf_interpolated)
            print(f"MPF data successfully interpolated and saved at {fn}.")
        except Exception as e:
            print(f"MPF data could not be interpolated: {e}")

        # predict MPF using constants provided by Landy et al, 2015
        pred_mpf = predict_mpf(MISR, R0, L, TAU, HNET)

        # to find the overlap - gives only the points that exist in both sets
        mpf_overlap = mpf_interpolated + (0 * pred_mpf)
        pred_overlap = pred_mpf + (0 * mpf_interpolated)

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

        # calculate r^2 (Pearson's product correlation coefficient) between true and predicted MPF
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
            fig_fn = f'/home/htweedie/melt_ponds/Istomina_MPF_figs/Istomina_pred/MPF_pred_{year}_2.png'
            plt.savefig(fig_fn, format='png')
            print(f"{year} figure saved at {fig_fn}")
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
