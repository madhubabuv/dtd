import numpy as np
import os
from abc import ABCMeta, abstractmethod
from .image import Image
import copy

class Capture(metaclass=ABCMeta):

    @abstractmethod
    def capture_image(self, camera, timestamp):
        pass
    
    def convert_raw_images(self, raw_images):
        pass



class Camera(Capture):

    def __init__(self, name, image_resolution,
                    focal_x, focal_y, principal_x, principal_y,distortion_coefficients = None):

        self.name = name
        self.image_resolution = image_resolution
        self.camera_matrix = self.to_camera_matrix(focal_x, focal_y, principal_x, principal_y)
        self.full_res_camera_matrix = copy.deepcopy(self.camera_matrix)#.copy()
        self.distortion_coefficients = distortion_coefficients
        self.working_resolution = None


    def to_camera_matrix(self, focal_x, focal_y, principal_x, principal_y):
        return np.array([[focal_x, 0, principal_x],
                         [0, focal_y, principal_y],
                         [0, 0, 1]])


    def set_working_resolution(self, resolution):

        orig_width, orig_height = self.image_resolution
        new_width, new_height = resolution

        if orig_width != new_width or orig_height != new_height:
            self.working_resolution = resolution
            self.camera_matrix[0, 0] = self.camera_matrix[0, 0] * new_width / orig_width
            self.camera_matrix[1, 1] = self.camera_matrix[1, 1] * new_height / orig_height
            self.camera_matrix[0, 2] = self.camera_matrix[0, 2] * new_width / orig_width
            self.camera_matrix[1, 2] = self.camera_matrix[1, 2] * new_height / orig_height

    def set_data_path(self, data_path):

        self.data_path = data_path

    def capture_image(self, timestamp,ext = '.png', undistort = False, full_resolution = False):

        image_path = os.path.join(self.data_path, self.name, 'data', str(timestamp) + ext)
        assert os.path.exists(image_path), "Image does not exist: {}".format(image_path)
        image = Image.read(image_path)
        full_res_image = image.copy()
        if undistort:
            image =  Image.undistort(image, self.camera_matrix, self.distortion_coefficients)

        if not self.working_resolution is None:
            image = Image.resize(image, self.working_resolution[0], self.working_resolution[1])
            
        # cv2 reads images in BGR format, so we convert it to RGB
        image = Image.to_rgb(image)
        image = Image.normalize(image) 
        
        if full_resolution:
            full_res_image = Image.to_rgb(full_res_image)
            full_res_image = Image.normalize(full_res_image)
            return (image, full_res_image)
        return image




        


class StereoCamera:

    def __init__(self, camera_rig_info):

        camera_0 = camera_rig_info['camera_0']
        camera_1 = camera_rig_info['camera_1']

        self.camera_0 = Camera(camera_0['name'], camera_0['resolution'],
                                camera_0['focal_x'], camera_0['focal_y'],
                                camera_0['principal_x'], camera_0['principal_y'],
                                camera_0['distortion_coefficients'])

        self.camera_1 = Camera(camera_1['name'], camera_1['resolution'],
                                camera_1['focal_x'], camera_1['focal_y'],
                                camera_1['principal_x'], camera_1['principal_y'],
                                camera_1['distortion_coefficients'])

        self.baseline_distance = camera_rig_info['baseline_distance']
            
    def set_data_path(self, data_path):

        self.data_path = data_path
        self.camera_0.set_data_path(data_path)
        self.camera_1.set_data_path(data_path)

    def capture_image(self, timestamp,ext = '.png', undistort = False, full_resolution = False):
        
        image_0 = self.camera_0.capture_image(timestamp,ext, undistort, full_resolution = full_resolution)
        image_1 = self.camera_1.capture_image(timestamp,ext, undistort, full_resolution = full_resolution)

        return image_0, image_1

    def set_working_resolution(self, resolution):

        self.camera_0.set_working_resolution(resolution)
        self.camera_1.set_working_resolution(resolution)



class MonoCamera:

    def __init__(self, camera_info):

        self.camera = Camera(camera_info['name'], camera_info['resolution'],
                                camera_info['focal_x'], camera_info['focal_y'],
                                camera_info['principal_x'], camera_info['principal_y'],
                                camera_info['distortion_coefficients'])

    def set_data_path(self, data_path):

        self.camera.set_data_path(data_path)

    def set_working_resolution(self, resolution):
        
        self.camera.set_working_resolution(resolution)

    def capture_image(self, timestamp,ext = '.png', undistort = False):

        return self.camera.capture_image(timestamp,ext, undistort)

    def undistort_image(self, image):

        return self.camera.undistort_image(image)