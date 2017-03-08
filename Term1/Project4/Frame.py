import numpy as np
import cv2
from pprint import pprint as pp
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

def PerspectiveTransform(plotfig=False):
    """
    Create and return the matrix M for perspective
    transformation of an image
    """
    mtx, dist = CalibrateCamera()
    img = mpimg.imread("test_images/straight_lines1.jpg")

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    offset = 100
    img_size = (gray.shape[1], gray.shape[0])
    src = np.float32([(275, 680),  (600, 447), (682, 447), (1040, 680)])
    src = np.float32([(220, 720),  (600, 447), (682, 447), (1110, 720)])
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])
    dst = np.float32([[350, img_size[1]], [350, 0], [980, 0], [980, img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)

    if plotfig:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 3])
        ax1.imshow(undist)
        ax1.plot(src[:,0], src[:,1], '-r')
        ax2.imshow(warped)
        ax2.plot(dst[:,0], dst[:,1], '-r')
        plt.show()

    return M

class Frame(object):
    """docstring for Frame."""
    def __init__(self, image, camera_cal=None, sobel_kernel=5, color_scheme='GRAY'):
        self.image = image
        if camera_cal is None:
            self.mtx, self.dist = CalibrateCamera()
        else:
            self.mtx, self.dist = camera_cal
        self.undrt_image = cv2.undistort(self.image, self.mtx, self.dist, None, self.mtx)


        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.sobelx = np.zeros_like(self.gray_image)
        self.sobely = np.zeros_like(self.gray_image)
        self.sobelx_thresh = np.zeros_like(self.gray_image)
        self.sobely_thresh = np.zeros_like(self.gray_image)
        self.grad_mag_thresh = np.zeros_like(self.gray_image)
        self.grad_dir_thresh = np.zeros_like(self.gray_image)

    def sobel(self, sobel_kernel=3):
        self.sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        self.sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    def abs_sobel_thresh(self, orient='x', thresh=(0, 255)):
        """
        Return an image after applying a threshold to either
        x or y component of the Sobel gradient
        """
        if orient == 'x':
            abs_sobel = np.absolute(self.sobelx)
        if orient == 'y':
            abs_sobel = np.absolute(self.sobely)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        if orient == 'x':
            self.sobelx_thresh = binary_output
        if orient == 'y':
            self.sobely_thresh = binary_output

    def mag_thresh(self, sobel_kernel=3, thresh=(90, 255)):
        """
        Return an image after applying a threshold to Sobel gradient magnitude
        """
        gradmag = np.sqrt(self.sobelx**2 + self.sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        self.grad_mag_threah = binary_output

    def dir_thresh(self, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        Return an image after applying a threshold to Sobel gradient direction
        """
        absgraddir = np.arctan2(np.absolute(self.sobely), np.absolute(self.sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        self.grad_dir_thresh = binary_output

    def grayscale(self):
        return self.grayscale

    def sobel_transform(self, sobel_kernel=3):
        sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        return (sobelx, sobely)

    def thresh_sobel_dir(img, thresh=(0, np.pi/2)):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return binary_output

    def hls_select(img, thresh=(0, 255), channel=0):
        """
        Select one channel of the HLS color scheme and return a binany image
        """
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        chan_im = hls[:,:,channel]
        binary_output = np.zeros_like(chan_im)
        binary_output[(chan_im > thresh[0]) & (chan_im <= thresh[1])] = 1

        return binary_output

    def hsv_select(img, thresh=(0, 255), channel=0):
        """
        Select one channel of the HSV color scheme and return a binany image
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        chan_im = hsv[:,:,channel]
        binary_output = np.zeros_like(chan_im)
        binary_output[(chan_im > thresh[0]) & (chan_im <= thresh[1])] = 1

        return binary_output

    def rgb_select(img, thresh=(0, 255), channel=0):
        """
        Select one channel of the RGB color scheme and return a binany image
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        chan_im = rgb[:,:,channel]
        binary_output = np.zeros_like(chan_im)
        binary_output[(chan_im > thresh[0]) & (chan_im <= thresh[1])] = 1

        return binary_output


class Line(object):
    """docstring for Line."""
    def __init__(self, arg):
        super(Line, self).__init__()
        self.arg = arg
