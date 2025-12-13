import sunpy.map
import numpy as np
from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a
import astropy.units as u
import os
import glob


def load_and_prepare_data(target_date=None):
    if target_date is not None:
        try:
            date_str = target_date.strftime('%Y-%m-%d %H:%M:%S')
            start_time = date_str
            end_time = (target_date + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')

            print(f"Searching for HMI data from {start_time} to {end_time}...")

            result = Fido.search(a.Time(start_time, end_time),
                                 a.Instrument('HMI'),
                                 a.Physobs('los_magnetic_field'),
                                 a.Sample(720 * u.s))

            print(f"Found {len(result[0])} files")

            if len(result[0]) > 0:
                sunpy_data_dir = "C:\\Users\\user\\sunpy\\data"
                if os.path.exists(sunpy_data_dir):
                    search_pattern = os.path.join(sunpy_data_dir, f"*{target_date.strftime('%Y.%m.%d')}*.fits")
                    existing_files = glob.glob(search_pattern)

                    if existing_files:
                        print(f"Found existing file in SunPy data directory: {existing_files[0]}")
                        sample_map = sunpy.map.Map(existing_files[0])
                        print(f"Loaded existing data for: {sample_map.date}")
                    else:
                        print("No existing files found, downloading...")
                        downloaded_files = Fido.fetch(result[0][0])
                        print(f"Downloaded: {downloaded_files}")

                        sample_map = sunpy.map.Map(downloaded_files[0])
                        print(f"Successfully loaded real data for: {sample_map.date}")
                else:
                    print("SunPy data directory not found, downloading...")
                    downloaded_files = Fido.fetch(result[0][0])
                    print(f"Downloaded: {downloaded_files}")

                    sample_map = sunpy.map.Map(downloaded_files[0])
                    print(f"Successfully loaded real data for: {sample_map.date}")

            else:
                print(f"No data found for {target_date}, using sample data")
                from sunpy.data.sample import HMI_LOS_IMAGE
                sample_map = sunpy.map.Map(HMI_LOS_IMAGE)

        except Exception as e:
            print(f"Error loading real data: {e}")
            print("Using sample data instead")
            from sunpy.data.sample import HMI_LOS_IMAGE
            sample_map = sunpy.map.Map(HMI_LOS_IMAGE)
    else:
        from sunpy.data.sample import HMI_LOS_IMAGE
        sample_map = sunpy.map.Map(HMI_LOS_IMAGE)

    print(f"Data shape: {sample_map.data.shape}")
    print(f"Observation date: {sample_map.date}")

    data = sample_map.data
    data_clean = np.nan_to_num(data, nan=0.0)

    sample_map_clean = sunpy.map.Map(data_clean, sample_map.meta)

    height, width = data_clean.shape
    reference_center = (width / 2, height / 2)

    return sample_map_clean, data_clean, reference_center
