import numpy as np


def calculate_edge_based_uncertainty(data_clean, center, disk_radius=400, edge_width=3):
    y_coords, x_coords = np.indices(data_clean.shape)
    distances = np.sqrt((x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2)

    edge_mask = (distances > disk_radius - edge_width) & (distances < disk_radius + edge_width)

    abs_data = np.abs(data_clean)

    if np.sum(edge_mask) == 0:
        return None

    threshold = np.percentile(abs_data[edge_mask], 70)
    bright_edge_mask = edge_mask & (abs_data > threshold)

    weights = abs_data * bright_edge_mask

    if np.sum(weights) == 0:
        return None

    dx = x_coords - center[0]
    dy = y_coords - center[1]

    m_00 = np.sum(weights)

    if m_00 == 0:
        return None

    mu_20 = np.sum(weights * dx ** 2) / m_00
    mu_02 = np.sum(weights * dy ** 2) / m_00
    mu_11 = np.sum(weights * dx * dy) / m_00

    std_x = np.sqrt(mu_20)
    std_y = np.sqrt(mu_02)

    correlation = mu_11 / (std_x * std_y) if std_x * std_y > 0 else 0

    lambda1 = 0.5 * (mu_20 + mu_02 + np.sqrt((mu_20 - mu_02) ** 2 + 4 * mu_11 ** 2))
    lambda2 = 0.5 * (mu_20 + mu_02 - np.sqrt((mu_20 - mu_02) ** 2 + 4 * mu_11 ** 2))

    if mu_20 != mu_02:
        ellipse_angle = 0.5 * np.arctan2(2 * mu_11, mu_20 - mu_02)
    else:
        ellipse_angle = np.pi / 4 if mu_11 > 0 else -np.pi / 4

    return {
        'std_pixels': (std_x, std_y),
        'variance_x': mu_20,
        'variance_y': mu_02,
        'covariance': mu_11,
        'correlation': correlation,
        'error_ellipse_major': np.sqrt(lambda1),
        'error_ellipse_minor': np.sqrt(lambda2),
        'error_ellipse_angle': ellipse_angle,
        'confidence_68': (std_x, std_y),
        'confidence_95': (2 * std_x, 2 * std_y),
        'edge_pixels_used': np.sum(bright_edge_mask),
        'total_edge_pixels': np.sum(edge_mask),
        'method': 'improved_edge_based'
    }
