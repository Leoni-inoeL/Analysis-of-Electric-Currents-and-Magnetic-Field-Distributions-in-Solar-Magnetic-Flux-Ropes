import sunpy.map
import numpy as np
from datetime import timedelta
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

    reference_center = None
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
                print(f"Using HMI_LOS_IMAGE sample data: {HMI_LOS_IMAGE}")

        except Exception as e:
            print(f"Error loading magnetogram data: {e}")
            print("Using sample magnetogram data instead")
            from sunpy.data.sample import HMI_LOS_IMAGE
            sample_map = sunpy.map.Map(HMI_LOS_IMAGE)
            print(f"Using HMI_LOS_IMAGE sample data: {HMI_LOS_IMAGE}")
    else:
        from sunpy.data.sample import HMI_LOS_IMAGE
        sample_map = sunpy.map.Map(HMI_LOS_IMAGE)
        print(f"Using HMI_LOS_IMAGE sample data: {HMI_LOS_IMAGE}")

    print(f"Magnetogram data shape: {sample_map.data.shape}")
    print(f"Magnetogram observation date: {sample_map.date}")

    data = sample_map.data
    data_clean = np.nan_to_num(data, nan=0.0)

    sample_map_clean = sunpy.map.Map(data_clean, sample_map.meta)

    magnetogram_center = None
    if 'CRPIX1' in sample_map.meta and 'CRPIX2' in sample_map.meta:
        magnetogram_center = (
            float(sample_map.meta['CRPIX1']) - 1,
            float(sample_map.meta['CRPIX2']) - 1
        )

    white_light_metadata_center = None
    if white_map is not None and 'CRPIX1' in white_map.meta and 'CRPIX2' in white_map.meta:
        white_light_metadata_center = (
            float(white_map.meta['CRPIX1']) - 1,
            float(white_map.meta['CRPIX2']) - 1
        )
        print(f"\nWhite light (0-based) center: ({white_light_metadata_center[0]:.2f}, "
              f"{white_light_metadata_center[1]:.2f})")

    if magnetogram_center is not None:
        reference_center = magnetogram_center
        print(f"\nUsing magnetogram reference center: {reference_center}")

        if white_light_metadata_center is not None:
            dx = white_light_metadata_center[0] - magnetogram_center[0]
            dy = white_light_metadata_center[1] - magnetogram_center[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            print(f"White light/magnetogram difference: Δx={dx:+.2f}, Δy={dy:+.2f}, dist={distance:.2f} px")

    elif white_light_metadata_center is not None:
        reference_center = white_light_metadata_center
        print(f"\nUsing white light as reference center: {reference_center}")

    return sample_map_clean, data_clean, reference_center
