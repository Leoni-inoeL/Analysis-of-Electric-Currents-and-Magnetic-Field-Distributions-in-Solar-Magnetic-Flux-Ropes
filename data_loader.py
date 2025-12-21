import sunpy.map
import numpy as np
from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a
import astropy.units as u
import os
import glob
from white_light_finder import get_white_light_center
from circle_bubbling_method import CircleBubblingData


def load_and_prepare_data(target_date=None):
    print("\nLoading white light data")
    white_light_center, white_map = get_white_light_center(target_date)

    if white_map is not None:
        white_data_clean = np.nan_to_num(white_map.data, nan=0.0)
        CircleBubblingData.set_white_light_data(white_data_clean)
        print(f"White light data stored for Circle Bubbling")

    if target_date is not None:
        try:
            date_str = target_date.strftime('%Y-%m-%d %H:%M:%S')
            start_time = date_str
            end_time = (target_date + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')

            print(f"Searching for HMI magnetogram data from {start_time} to {end_time}...")

            result = Fido.search(a.Time(start_time, end_time),
                                 a.Instrument('HMI'),
                                 a.Physobs('los_magnetic_field'),
                                 a.Sample(720 * u.s))

            print(f"Found {len(result[0])} files")

            if len(result[0]) > 0:
                sunpy_data_dir = "C:\\Users\\user\\sunpy\\data"
                if os.path.exists(sunpy_data_dir):
                    search_pattern = os.path.join(sunpy_data_dir,
                                                  f"*{target_date.strftime('%Y.%m.%d')}*magnetogram*.fits")
                    existing_files = glob.glob(search_pattern)

                    if existing_files:
                        print(f"Found existing magnetogram file: {existing_files[0]}")
                        sample_map = sunpy.map.Map(existing_files[0])
                        print(f"Loaded magnetogram data for: {sample_map.date}")
                    else:
                        print("No existing magnetogram files found, downloading...")
                        downloaded_files = Fido.fetch(result[0][0])
                        sample_map = sunpy.map.Map(downloaded_files[0])
                        print(f"Successfully loaded magnetogram data for: {sample_map.date}")
                else:
                    print("SunPy data directory not found, downloading magnetogram...")
                    downloaded_files = Fido.fetch(result[0][0])
                    sample_map = sunpy.map.Map(downloaded_files[0])
                    print(f"Successfully loaded magnetogram data for: {sample_map.date}")

            else:
                print(f"No magnetogram data found for {target_date}, using sample data")
                from sunpy.data.sample import HMI_LOS_IMAGE
                sample_map = sunpy.map.Map(HMI_LOS_IMAGE)

        except Exception as e:
            print(f"Error loading magnetogram data: {e}")
            print("Using sample magnetogram data instead")
            from sunpy.data.sample import HMI_LOS_IMAGE
            sample_map = sunpy.map.Map(HMI_LOS_IMAGE)
    else:
        from sunpy.data.sample import HMI_LOS_IMAGE
        sample_map = sunpy.map.Map(HMI_LOS_IMAGE)

    print(f"Magnetogram data shape: {sample_map.data.shape}")
    print(f"Magnetogram observation date: {sample_map.date}")

    data = sample_map.data
    data_clean = np.nan_to_num(data, nan=0.0)

    sample_map_clean = sunpy.map.Map(data_clean, sample_map.meta)

    if white_light_center:
        reference_center = white_light_center
        print(f"Using white light center as reference: ({reference_center[0]:.2f}, {reference_center[1]:.2f})")
    else:
        height, width = data_clean.shape
        reference_center = (width / 2, height / 2)
        print(f"Using image center as reference: ({reference_center[0]:.2f}, {reference_center[1]:.2f})")

    return sample_map_clean, data_clean, reference_center
