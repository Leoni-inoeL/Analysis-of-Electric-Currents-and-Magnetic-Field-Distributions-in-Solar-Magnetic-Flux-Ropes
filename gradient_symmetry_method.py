import numpy as np
from uncertainty import calculate_edge_based_uncertainty


def gradient_symmetry(data_clean):
    try:
        grad_y, grad_x = np.gradient(data_clean)
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_threshold = np.percentile(grad_magnitude, 70)
        significant_grad = grad_magnitude > grad_threshold

        if np.any(significant_grad):
            y_coords, x_coords = np.where(significant_grad)
            weights = grad_magnitude[significant_grad]
            center_x = np.average(x_coords, weights=weights)
            center_y = np.average(y_coords, weights=weights)
        else:
            x_weights = np.sum(np.abs(grad_x), axis=0)
            y_weights = np.sum(np.abs(grad_y), axis=1)
            center_x = np.average(np.arange(data_clean.shape[1]), weights=x_weights)
            center_y = np.average(np.arange(data_clean.shape[0]), weights=y_weights)

        center = (center_x, center_y)
        uncertainty = calculate_edge_based_uncertainty(data_clean, center)

        return (center_x, center_y), "Gradient Symmetry", uncertainty

    except Exception as e:
        print(f"Gradient method failed: {e}")
        return None, "Gradient Symmetry", None
