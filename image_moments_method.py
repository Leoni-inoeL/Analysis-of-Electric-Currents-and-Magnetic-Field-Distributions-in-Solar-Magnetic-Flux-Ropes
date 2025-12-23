import numpy as np
from uncertainty import calculate_edge_based_uncertainty
from hmi_processor import HMI_Processor


class ImageMomentsMethod(HMI_Processor):
    def __init__(self):
        super().__init__()
        
    def solar_center(self):
        if self.data is None:
            print("No data loaded. Call read_fits() first.")
            return None, None, None
        
        return self.process_method(self.data)
    
    def process_method(self, data_clean):
        try:
            data_squared = data_clean ** 2
            y_coords, x_coords = np.indices(data_squared.shape)

            m_00 = np.sum(data_squared)
            if m_00 == 0:
                return None, "Image Moments", None

            m_10 = np.sum(x_coords * data_squared)
            m_01 = np.sum(y_coords * data_squared)

            cx = m_10 / m_00
            cy = m_01 / m_00

            center = (cx, cy)
            uncertainty = calculate_edge_based_uncertainty(data_clean, center)

            return (cx, cy), "Image Moments", uncertainty

        except Exception as e:
            print(f"Moments analysis failed: {e}")
            return None, "Image Moments", None


def moments_analysis(data_clean):
    processor = ImageMomentsMethod()
    processor.data = data_clean
    return processor.solar_center()
