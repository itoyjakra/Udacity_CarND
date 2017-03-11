import numpy as np
import cv2
from pprint import pprint as pp
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Frame(object):
    """docstring for Frame."""
    def __init__(self, image, camera_cal=None, sobel_kernel=5, color_scheme='GRAY'):
        self.image = image

    def sobel(self, sobel_kernel=3):
        """
        Calculates the sobel gradients
        """
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        return (sobelx, sobely)

    def sobel_thresh(self, sobel_kernel=3, orient='x', thresh=(0, 255)):
        """
        Return an image after applying a threshold to either
        x or y component of the Sobel gradient
        """
        sobelx, sobely = self.sobel(sobel_kernel=sobel_kernel)
        if orient == 'x':
            abs_sobel = np.absolute(sobelx)
        if orient == 'y':
            abs_sobel = np.absolute(sobely)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def mag_thresh(self, sobel_kernel=3, thresh=(90, 255)):
        """
        Return an image after applying a threshold to Sobel gradient magnitude
        """
        sobelx, sobely = self.sobel(sobel_kernel=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        return binary_output

    def dir_thresh(self, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        Return an image after applying a threshold to Sobel gradient direction
        """
        sobelx, sobely = self.sobel(sobel_kernel=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return binary_output

    def grayscale(self):
        return self.grayscale

    def hls_select(self, thresh=(0, 255), channel=0):
        """
        Select one channel of the HLS color scheme and return a binany image
        """
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
        chan_im = hls[:,:,channel]
        binary_output = np.zeros_like(chan_im)
        binary_output[(chan_im > thresh[0]) & (chan_im <= thresh[1])] = 1

        return binary_output

    def hsv_select(self, thresh=(0, 255), channel=0):
        """
        Select one channel of the HSV color scheme and return a binany image
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        chan_im = hsv[:,:,channel]
        binary_output = np.zeros_like(chan_im)
        binary_output[(chan_im > thresh[0]) & (chan_im <= thresh[1])] = 1

        return binary_output

    def rgb_select(self, thresh=(0, 255), channel=0):
        """
        Select one channel of the RGB color scheme and return a binany image
        """
        rgb = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        chan_im = rgb[:,:,channel]
        binary_output = np.zeros_like(chan_im)
        binary_output[(chan_im > thresh[0]) & (chan_im <= thresh[1])] = 1

        return binary_output

    def process(self, M_perp_tran, plotfig=False):
        """
        Pipeline to apply a sequence of steps to process an
        image and return a warped image ready to be consumed
        by the Lane class
        """
        # apply color channel thresholds
        ch1 = self.hsv_select(thresh=(150, 255), channel=1)
        ch2 = self.hsv_select(thresh=(200, 255), channel=2)
        ch3 = self.rgb_select(thresh=(0, 30))

        # apply Sobel gradient thresholds
        dir_binary = self.dir_thresh(sobel_kernel=15, thresh=(0.5, 1.1))
        mag_binary = self.mag_thresh(sobel_kernel=15, thresh=(20, 125))
        sobel_binary = self.sobel_thresh(sobel_kernel=15, orient='x', thresh=(200, 255))

        # combine filters
        combined = np.zeros_like(dir_binary)
        combined[(mag_binary == 1) & (dir_binary == 1) | (sobel_binary == 1)] = 1
        combined = combined.astype(np.uint8)
        ch = ((ch1 | ch2) & ch3) | combined

        if plotfig:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[24, 6])
            ax1.imshow(self.image)
            ax2.imshow(ch, cmap='gray')
            plt.show()

        # warp image for lane detection
        img_size = (self.image.shape[1], self.image.shape[0])
        warped = cv2.warpPerspective(ch, M_perp_tran, img_size)

        if plotfig:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[24, 6])
            ax1.imshow(ch, cmap='gray')
            ax2.imshow(warped, cmap='gray')
            plt.show()

        return warped
