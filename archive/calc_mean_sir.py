import pandas as pd
import numpy as np
from datetime import datetime


def main():
    YEAR = 2019
    SPACING = 8
    DAYS_TO_FORWARD = 183

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'Time started: {current_time}')

    sir_advected = pd.read_pickle(f'/home/htweedie/melt_ponds/data/forwarded_mpfs/testing/sir_from_{YEAR}0401_{DAYS_TO_FORWARD}_days_spacing_{SPACING}.pkl')

    date_from = format_date(YEAR, '04', '01')
    date_to = format_date(YEAR, '08', '31')

    subset_sir = sir_advected.loc[date_from:date_to]
    num_points = subset_sir.shape[1]

    mean_sir = np.zeros(num_points)
    for i in range(num_points):
        if i % 10000 == 0:
            print(f'Processing grid cell {i} of {num_points}')
        timeseries = np.asarray(subset_sir[i])
        mean_sir[i] = np.nanmean(timeseries)

    fn = f'/home/htweedie/melt_ponds/data/forwarded_mpfs/testing/mean_summer_sir_{YEAR}'
    np.save(fn, mean_sir)
    print(f'Data saved at {fn}')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'Time finished: {current_time}')


def format_date(year, month, day):
    return f"{year}-{month}-{day} 12:00:00"


if __name__ == "__main__":
    main()
