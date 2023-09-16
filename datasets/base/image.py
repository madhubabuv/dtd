import numpy as np
import cv2 


class Image:

    def __init__(self):

        pass
    
    @staticmethod
    def read( path):
        return cv2.imread(path)

    @staticmethod
    def write( path, image):
        cv2.imwrite(path, image)

    @staticmethod
    def resize( image, width, height):
        return cv2.resize(image, (width, height))

    @staticmethod
    def undistort(img, K, dist_coef):
        
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dist_coef, (w, h), 0, (w, h))
        undistorted_img = cv2.undistort(img, K, dist_coef, None, newcameramtx)
        x, y, w, h = roi
        undistorted_img = undistorted_img[y:y + h, x:x + w]

        return undistorted_img,newcameramtx

    @staticmethod
    def to_grayscale( image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_rgb( image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def normalize( image):
        return image / 255.0

    @staticmethod
    def denormalize( image):

        image = image * 255.0
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        return image


    @staticmethod
    def to_float( image):
        return image.astype(np.float32)


    