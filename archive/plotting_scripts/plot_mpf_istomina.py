import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

def main():
    # get available years from directory
    years = listdir(f'/home/htweedie/melt_ponds/data/OLCI/olci')
    years.remove('LongitudeLatitudeGrid-n12500-Arctic.h5')
    years.remove('readme.txt')

    # set momths to retrieve and plot
    months = ['06', '07', '08']

    for year in years:
        monthly_mpfs = []

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
                    fn = f'/home/htweedie/melt_ponds/data/OLCI/olci/{year}/data/mpd1_{year}{month}{day}.nc'
                    ds = xr.open_dataset(fn)
                    mpf = ds['mpf']
                    daily_mpfs.append(np.array(mpf))
                except Exception as e:
                    print(f'Data could not be retrieved for {day}/{month}/{year}: {e}')
                    continue

            monthly_mpfs.append(np.nanmean(daily_mpfs, 0))

        # find mean of all months
        mean_summer_mpf = np.nanmean(monthly_mpfs, 0)

        # plot each month and the mean
        fig = plt.figure(figsize=(10,10))

        plt.subplot(221)
        plt.pcolormesh(monthly_mpfs[0][150:650])
        plt.title(f'Observed MPF June {year}')

        plt.subplot(222)
        plt.pcolormesh(monthly_mpfs[1][150:650])
        plt.title(f'Observed MPF July {year}')

        plt.subplot(223)
        plt.pcolormesh(monthly_mpfs[2][150:650])
        plt.title(f'Observed MPF August {year}')

        plt.subplot(224)
        plt.pcolormesh(mean_summer_mpf[150:650])
        plt.title(f'Observed MPF Mean {year}')

        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        cax = plt.axes((0.85, 0.1, 0.075, 0.8))
        plt.colorbar(cax=cax)

        fig_fn = f'/home/htweedie/melt_ponds/Istomina_MPF_figs/MPF_{year}.png'
        plt.savefig(fig_fn, format='png')
        print(f'Plot for {year} saved at {fig_fn}.')


if __name__ == '__main__':
    main()
