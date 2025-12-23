import numpy as np
import sunpy.map
from abc import ABC, abstractmethod


class HMI_Processor(ABC):
    def __init__(self):
        self.data = None
        self.metadata = None
        
    def read_fits(self, filepath):
        try:
            map_data = sunpy.map.Map(filepath)
            self.data = np.nan_to_num(map_data.data, nan=0.0)
            self.metadata = map_data.meta
            print(f"Loaded data from {filepath}")
            print(f"Data shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error reading FITS file: {e}")
            return None
    
    def solar_center(self):
        if self.data is None:
            print("No data loaded. Call read_fits() first.")
            return None, None
        
        height, width = self.data.shape
        center = (width / 2, height / 2)
        
        return center, "Base HMI Processor"
    
    @abstractmethod
    def process_method(self, data_clean):
        pass
