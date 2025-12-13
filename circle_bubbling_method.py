import numpy as np
import math


def generate_directions(n_directions=8):
    directions = []
    for i in range(n_directions):
        angle = 2 * math.pi * i / n_directions
        dx = math.cos(angle)
        dy = math.sin(angle)
        directions.append((dx, dy))
    return directions


def circle_brightness(image, center, diameter, n_points=360):
    radius = diameter / 2
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    brightness_values = []

    for angle in angles:
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)

        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = int(np.ceil(x)), int(np.ceil(y))

        if (0 <= x0 < image.shape[1] and 0 <= x1 < image.shape[1] and
                0 <= y0 < image.shape[0] and 0 <= y1 < image.shape[0]):
            dx, dy = x - x0, y - y0
            value = (image[y0, x0] * (1 - dx) * (1 - dy) +
                     image[y0, x1] * dx * (1 - dy) +
                     image[y1, x0] * (1 - dx) * dy +
                     image[y1, x1] * dx * dy)

            brightness_values.append(abs(value))

    if len(brightness_values) < n_points * 0.7:
        return 0

    return np.mean(brightness_values)


def circle_bubbling_algorithm(data_clean, initial_center=None, initial_diameter=None, n_points=180):
    height, width = data_clean.shape

    if initial_center is None:
        initial_center = (width / 2, height / 2)

    if initial_diameter is None:
        initial_diameter = min(width, height) * 0.6

    current_center = [float(initial_center[0]), float(initial_center[1])]
    current_diameter = float(initial_diameter)
    dd = current_diameter / 4

    directions = generate_directions(8)

    iteration = 0
    max_iterations = 50

    print(f"Start params: center=({current_center[0]:.1f}, {current_center[1]:.1f}), "
          f"diameter={current_diameter:.1f}, dD={dd:.1f}")

    while dd > 0.5 and iteration < max_iterations:
        iteration += 1
        improved = False

        current_brightness = circle_brightness(data_clean, current_center, current_diameter, n_points)

        for dx, dy in directions:
            new_center = [
                current_center[0] + dx * dd,
                current_center[1] + dy * dd
            ]

            new_diameter = current_diameter + dd
            new_brightness = circle_brightness(data_clean, new_center, new_diameter, n_points)

            if new_brightness > current_brightness:
                current_center = new_center
                current_diameter = new_diameter
                improved = True
                break

        if not improved:
            dd = dd / 2

    print(f"{iteration} iterations in total")
    print(f"Final center: ({current_center[0]:.2f}, {current_center[1]:.2f})")
    print(f"Final D: {current_diameter:.2f}")

    return tuple(current_center), current_diameter


def circle_bubbling_method(data_clean):
    try:
        abs_data = np.abs(data_clean)

        threshold = np.percentile(abs_data, 80)
        mask = abs_data > threshold
        y_indices, x_indices = np.where(mask)

        if len(x_indices) == 0:
            return None, "Circle Bubbling", None

        initial_center_x = np.mean(x_indices)
        initial_center_y = np.mean(y_indices)
        initial_center = (initial_center_x, initial_center_y)

        if len(x_indices) > 0:
            initial_diameter = (np.max(x_indices) - np.min(x_indices) + np.max(y_indices) - np.min(y_indices)) / 2
        else:
            initial_diameter = min(data_clean.shape) * 0.5

        center, final_diameter = circle_bubbling_algorithm(
            abs_data, initial_center, initial_diameter
        )

        uncertainty = {
            'std_pixels': (2.0, 2.0),
            'final_diameter': final_diameter,
            'method': 'circle_bubbling'
        }

        return center, "Circle Bubbling", uncertainty

    except Exception as e:
        print(f"Circle bubbling method failed: {e}")
        import traceback
        traceback.print_exc()
        return None, "Circle Bubbling", None
