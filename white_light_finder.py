import numpy as np
from datetime import datetime, timedelta
from sunpy.net import Fido, attrs as a
import astropy.units as u
import sunpy.map
import os
import glob


def get_white_light_center(target_date):
    date_str = target_date.strftime('%Y-%m-%d %H:%M:%S')
    start_time = date_str
    end_time = (target_date + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')

    print(f"\nSearching for HMI white light data...")
    print(f"Time range: {start_time} to {end_time}")

    try:
        result = Fido.search(a.Time(start_time, end_time),
                             a.Instrument('HMI'),
                             a.Physobs('intensity'),
                             a.Sample(720 * u.s))

        print(f"Found white light files: {len(result[0])}")

        if len(result[0]) == 0:
            print("No white light data found, trying intensitygram...")
            result = Fido.search(a.Time(start_time, end_time),
                                 a.Instrument('HMI'),
                                 a.Physobs('intensitygram'),
                                 a.Sample(720 * u.s))
            print(f"Found intensitygram files: {len(result[0])}")

        if len(result[0]) > 0:
            sunpy_data_dir = "C:\\Users\\user\\sunpy\\data"
            if os.path.exists(sunpy_data_dir):
                patterns = [
                    f"*{target_date.strftime('%Y.%m.%d')}*intensity*.fits",
                    f"*{target_date.strftime('%Y%m%d')}*Ic*.fits",
                    f"*{target_date.strftime('%Y_%m_%d')}*continuum*.fits"
                ]

                white_light_file = None
                for pattern in patterns:
                    search_pattern = os.path.join(sunpy_data_dir, pattern)
                    files = glob.glob(search_pattern)
                    if files:
                        white_light_file = files[0]
                        print(f"Found white light file: {os.path.basename(white_light_file)}")
                        break

                if not white_light_file:
                    print("Downloading white light data...")
                    downloaded = Fido.fetch(result[0][0])
                    white_light_file = downloaded[0]
            else:
                print("Downloading white light data...")
                downloaded = Fido.fetch(result[0][0])
                white_light_file = downloaded[0]

            white_map = sunpy.map.Map(white_light_file)
            print(f"Loaded white light: {white_map.date}")
            print(f"Size: {white_map.data.shape}")

            data = np.nan_to_num(white_map.data, nan=0.0)
            threshold = np.percentile(data, 90)
            mask = data > threshold

            if np.sum(mask) > 1000:
                y_idx, x_idx = np.where(mask)
                center_x = np.mean(x_idx)
                center_y = np.mean(y_idx)
                print(f"Center (threshold+mass): ({center_x:.2f}, {center_y:.2f})")
            else:
                height, width = data.shape
                center_x, center_y = width / 2, height / 2
                print(f"Center (image center): ({center_x:.2f}, {center_y:.2f})")

            return (center_x, center_y), white_map

        else:
            print("No white light data available for this date")
            return None, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None


def simple_white_light_center(image_data):
    data_clean = np.nan_to_num(image_data, nan=0.0)
    threshold = np.percentile(data_clean, 92)
    mask = data_clean > threshold

    if np.sum(mask) == 0:
        height, width = data_clean.shape
        return width / 2, height / 2

    y_indices, x_indices = np.where(mask)
    center_x = np.mean(x_indices)
    center_y = np.mean(y_indices)

    return center_x, center_y
