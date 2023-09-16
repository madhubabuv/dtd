import numpy as np
import os

from abc import ABCMeta, abstractmethod

class Capture(metaclass=ABCMeta):

    @abstractmethod
    def capture_scan(self, camera, timestamp):
        pass
    
    def to_camera(self, scan, Rt):
        pass


class Lidar(Capture):

    def __init__(self, name,):

        self.name = name



    def set_data_path(self, data_path):

        self.data_path = data_path

    def capture_scan(self, timestamp, ext = '.bin'):


        scan_path = os.path.join(self.data_path, self.name, 'data', str(timestamp) + ext)
        points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # homogeneous

        return points


    def to_camera(self, scan, Rt):

        assert scan.shape[1] == 4, "Scan is not in homogeneous form!"

        transformed_pts = np.dot(Rt, scan.T).T

        return transformed_pts[:, :3]


    

    


    





