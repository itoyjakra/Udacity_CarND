import numpy as np
import cv2
from pprint import pprint as pp
import glob

def CalibrateCamera(glob_files='camera_cal/calibration*.jpg'):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(glob_files)

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # calibrate camera using the object and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    assert ret

    return mtx, dist

class Frame(object):
    """docstring for Frame."""
    def __init__(self, image, mtx, dist):
        self.image = image
        self.mtx = mtx
        self.dist = dist
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.undrt_image = cv2.undistort(self.image, self.mtx, self.dist, None, self.mtx)

    def __init__(self, image):
        self.image = image
        self.mtx, self.dist = CalibrateCamera()
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.undrt_image = cv2.undistort(self.image, self.mtx, self.dist, None, self.mtx)

    def shape(self):
        return image.shape

    def image(self):
        return self.image

    def grayscale(self):
        return self.gray_image

    def undistorted(self):
        return self.undrt_image

class Line(object):
    """docstring for Line."""
    def __init__(self, arg):
        super(Line, self).__init__()
        self.arg = arg
