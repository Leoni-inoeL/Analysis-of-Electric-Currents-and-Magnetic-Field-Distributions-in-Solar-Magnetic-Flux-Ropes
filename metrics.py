import numpy as np


def calculate_metrics(detected_center, reference_center):
    if detected_center is None:
        return None, None, None

    dx = detected_center[0] - reference_center[0]
    dy = detected_center[1] - reference_center[1]
    distance_error = np.sqrt(dx ** 2 + dy ** 2)

    return distance_error, dx, dy


def calculate_average_center(results_df_, reference_center):
    successful_methods = results_df_[results_df_['Error_pixels'].notna()]

    if len(successful_methods) == 0:
        return None

    avg_center_x = np.mean(successful_methods['Center_X'])
    avg_center_y = np.mean(successful_methods['Center_Y'])
    avg_center = (avg_center_x, avg_center_y)

    dx = avg_center_x - reference_center[0]
    dy = avg_center_y - reference_center[1]
    avg_error = np.sqrt(dx ** 2 + dy ** 2)

    weights = 1 / successful_methods['Error_pixels']
    weighted_center_x = np.average(successful_methods['Center_X'], weights=weights)
    weighted_center_y = np.average(successful_methods['Center_Y'], weights=weights)
    weighted_center = (weighted_center_x, weighted_center_y)

    w_dx = weighted_center_x - reference_center[0]
    w_dy = weighted_center_y - reference_center[1]
    weighted_error = np.sqrt(w_dx ** 2 + w_dy ** 2)

    return {
        'simple_average': {
            'center': avg_center,
            'error': avg_error,
            'delta_x': dx,
            'delta_y': dy
        },
        'weighted_average': {
            'center': weighted_center,
            'error': weighted_error,
            'delta_x': w_dx,
            'delta_y': w_dy
        },
        'methods_used': len(successful_methods)
    }
