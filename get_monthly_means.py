'''
This script calculates mean melt-pond fractions for the Arctic region for all years for which data are avilable.
The path to the data and the months to be averaged are entered as command-line arguments.

Heather Tweedie  
May 2024  
MSci Dissertation
'''

import xarray as xr
import numpy as np
from os import listdir
from sys import argv


def main(argv):

    if len(argv) < 2:
        print('Usage:')
        print('     get_monthly_means.py [directory path] [list of months to calculate]')
        print('     For example: get_monthly_means.py /home/htweedie/melt_ponds/data/OLCI/olci 06 07 08')
        return

    # set months defined in command line
    months = argv[2:]
    if len(months) == 0:
        print('Please give at least one month for which the mean will be calculated.')
        return
    print(f'Months to average: {months}')

    # get available years in directory
    dir_path = argv[1]
    years = listdir(dir_path)
    print(f'Available years: {years}')

    for year in years:

        print(f"======== Processing {year} ========")

        print(f"Retrieving data and calculating monthly means:")
        for month in months:
            # set the number of days for the given month
            match month:
                case '05' | '07' | '08':
                    days = 31
                case '06' | '09':
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
                    fn = f'{dir_path}/{year}/data/mpd1_{year}{month}{day}.nc'
                    ds = xr.open_dataset(fn)
                    mpf = ds['mpf']
                    daily_mpfs.append(np.array(mpf))
                except Exception as e:
                    print(f'Data could not be retrieved for {day}/{month}/{year}: {e}')
                    continue

            # calculate the mean of the current month and save as .npy
            month_mean = np.nanmean(daily_mpfs, 0)
            fn = f'{dir_path}/{year}/mean_{month}_{year}.npy'
            np.save(fn, month_mean)
            print(f"{month} mean calculated and saved at {fn}.")


if __name__ == '__main__':
    main(argv)
