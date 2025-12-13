import numpy as np
from uncertainty import calculate_edge_based_uncertainty


def center_of_mass(data_clean):
    abs_data = np.abs(data_clean)
    threshold = np.percentile(abs_data, 80)
    mask = abs_data > threshold

    y_indices, x_indices = np.where(mask)
    if len(x_indices) > 0:
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)

        center = (center_x, center_y)
        uncertainty = calculate_edge_based_uncertainty(data_clean, center)

        return (center_x, center_y), "Center of Mass", uncertainty

    return None, "Center of Mass", None
