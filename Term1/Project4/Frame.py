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
    def __init__(self, image, camera_cal=None):
        self.image = image
        if camera_cal is None:
            self.mtx, self.dist = CalibrateCamera()
        else:
            self.mtx, self.dist = camera_cal
        self.undrt_image = cv2.undistort(self.image, self.mtx, self.dist, None, self.mtx)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.sobel = np.zeros_like(self.gray_image)
        self.grad_mag_thresh = np.zeros_like(self.gray_image)
        self.grad_dir_thresh = np.zeros_like(self.gray_image)

    def mag_thresh(self, sobel_kernel=3, mag_thresh=(90, 255)):
        """
        Return an image after applying a threshold to Sobel gradient magnitude
        """
        sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        self.grad_mag_threah = binary_output

    def dir_thresh(self, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        Return an image after applying a threshold to Sobel gradient direction
        """
        absgraddir = np.arctan2(np.absolute(self.sobely), np.absolute(self.sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        self.grad_dir_thresh = binary_output

class Line(object):
    """docstring for Line."""
    def __init__(self, arg):
        super(Line, self).__init__()
        self.arg = arg
