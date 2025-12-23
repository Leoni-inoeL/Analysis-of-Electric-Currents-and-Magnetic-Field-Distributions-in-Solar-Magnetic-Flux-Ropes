import numpy as np
import math
from scipy.ndimage import gaussian_filter, sobel, minimum_filter, maximum_filter
from uncertainty import calculate_edge_based_uncertainty
from hmi_processor import HMI_Processor


class CircleBubblingData:
    white_light_data = None

    @classmethod
    def set_white_light_data(cls, white_data):
        cls.white_light_data = white_data

    @classmethod
    def get_white_light_data(cls):
        return cls.white_light_data


def apply_article_filters(image):
    filtered = gaussian_filter(image, sigma=1.0)
    sobel_x = sobel(filtered, axis=0)
    sobel_y = sobel(filtered, axis=1)
    filtered = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    local_min = minimum_filter(filtered, size=3)
    local_max = maximum_filter(filtered, size=3)
    mask_low = filtered < local_min
    mask_high = filtered > local_max
    filtered[mask_low] = local_min[mask_low]
    filtered[mask_high] = local_max[mask_high]
    return filtered


def circle_brightness_sum(image, center, diameter, n_points):
    radius = diameter / 2
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    total_brightness = 0.0
    valid_points = 0

    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        x_0, y_0 = int(np.floor(x)), int(np.floor(y))
        x_1, y_1 = int(np.ceil(x)), int(np.ceil(y))

        if (0 <= x_0 < image.shape[1] and 0 <= x_1 < image.shape[1] and
                0 <= y_0 < image.shape[0] and 0 <= y_1 < image.shape[0]):
            dx, dy = x - x_0, y - y_0
            value = (image[y_0, x_0] * (1 - dx) * (1 - dy) +
                     image[y_0, x_1] * dx * (1 - dy) +
                     image[y_1, x_0] * (1 - dx) * dy +
                     image[y_1, x_1] * dx * dy)
            total_brightness += abs(value)
            valid_points += 1

    if valid_points < n_points * 0.7:
        return 0.0

    return total_brightness


def circle_bubbling_algorithm(data_clean, initial_center, initial_diameter):
    height, width = data_clean.shape
    current_center = [float(initial_center[0]), float(initial_center[1])]
    current_diameter = float(initial_diameter)

    dd = 150.0

    directions = []
    for i in range(8):
        angle = math.pi * i / 4
        directions.append((math.cos(angle), math.sin(angle)))

    iteration = 0
    max_iterations = 200

    n_points = 600

    best_center = current_center[:]
    best_diameter = current_diameter
    best_brightness = circle_brightness_sum(data_clean, current_center, current_diameter, n_points)

    print(f"Start: center=({current_center[0]:.1f}, {current_center[1]:.1f}), D={current_diameter:.1f}, step={dd:.1f}")

    while dd > 0.05 and iteration < max_iterations:
        iteration += 1
        improved = False

        for dx, dy in directions:
            new_center = [
                current_center[0] + dx * dd,
                current_center[1] + dy * dd
            ]

            for diameter_change in [dd, -dd, dd / 2, -dd / 2, 0]:
                new_diameter = current_diameter + diameter_change

                if new_diameter < 100 or new_diameter > min(height, width) * 1.1:
                    continue

                new_brightness = circle_brightness_sum(data_clean, new_center, new_diameter, n_points)

                if new_brightness > best_brightness:
                    best_center = new_center[:]
                    best_diameter = new_diameter
                    best_brightness = new_brightness
                    current_center = new_center[:]
                    current_diameter = new_diameter
                    improved = True

                    if iteration <= 20:
                        print(
                            f"  Iter {iteration}: ({current_center[0]:.1f}, {current_center[1]:.1f}), D={current_diameter:.1f}")
                    break

            if improved:
                break

        if not improved:
            dd = dd * 0.6

    print(f"Iterations: {iteration}")
    print(f"Final center: ({best_center[0]:.2f}, {best_center[1]:.2f})")
    print(f"Final diameter: {best_diameter:.2f} px")
    print(f"Brightness: {best_brightness:.0f}")

    return tuple(best_center), best_diameter


class CircleBubblingMethod(HMI_Processor):
    def __init__(self):
        super().__init__()
        
    def solar_center(self):
        if self.data is None:
            print("No data loaded. Call read_fits() first.")
            return None, None, None
        
        return self.process_method(self.data)
    
    def process_method(self, data_clean):
        try:
            white_data = CircleBubblingData.get_white_light_data()

            if white_data is not None:
                print("Circle Bubbling: using WHITE LIGHT data")
                filtered_data = apply_article_filters(white_data)
                height, width = filtered_data.shape
                initial_center = (width / 2, height / 2)

                threshold = np.percentile(filtered_data, 80)
                mask = filtered_data > threshold

                if np.sum(mask) > 1000:
                    y_idx, x_idx = np.where(mask)
                    if len(x_idx) > 0:
                        initial_diameter = (np.max(x_idx) - np.min(x_idx) +
                                            np.max(y_idx) - np.min(y_idx)) / 2
                    else:
                        initial_diameter = min(height, width) * 0.8
                else:
                    initial_diameter = min(height, width) * 0.8

                print(f"Start from center: {initial_center}")
                print(f"Initial diameter: {initial_diameter:.1f} px")

                center, final_diameter = circle_bubbling_algorithm(
                    filtered_data, initial_center, initial_diameter
                )

                method_name = "Circle Bubbling (white light)"
                data_source = "white light"
                uncertainty_data = white_data

            else:
                print("Circle Bubbling: using magnetogram data")
                height, width = data_clean.shape
                initial_center = (width / 2, height / 2)
                working_data = np.abs(data_clean)
                threshold = np.percentile(working_data, 80)
                mask = working_data > threshold

                if len(mask) == 0 or np.sum(mask) == 0:
                    return None, "Circle Bubbling", None

                y_indices, x_indices = np.where(mask)

                if len(x_indices) > 0:
                    initial_diameter = (np.max(x_indices) - np.min(x_indices) +
                                        np.max(y_indices) - np.min(y_indices)) / 2
                else:
                    initial_diameter = min(data_clean.shape) * 0.5

                center, final_diameter = circle_bubbling_algorithm(
                    working_data, initial_center, initial_diameter
                )

                method_name = "Circle Bubbling (magnetogram)"
                data_source = "magnetogram"
                uncertainty_data = data_clean

            uncertainty = calculate_edge_based_uncertainty(
                uncertainty_data,
                center,
                disk_radius=final_diameter / 2,
                edge_width=10
            )

            if uncertainty:
                uncertainty['final_diameter'] = final_diameter
                uncertainty['data_source'] = data_source
                uncertainty['method'] = 'circle_bubbling'
            else:
                uncertainty = {
                    'std_pixels': (max(1.0, final_diameter * 0.005),
                                   max(1.0, final_diameter * 0.005)),
                    'final_diameter': final_diameter,
                    'data_source': data_source,
                    'method': 'circle_bubbling'
                }

            return center, method_name, uncertainty

        except Exception as e:
            print(f"Circle bubbling method failed: {e}")
            import traceback
            traceback.print_exc()
            return None, "Circle Bubbling", None


def circle_bubbling_method(data_clean):
    processor = CircleBubblingMethod()
    processor.data = data_clean
    return processor.solar_center()
