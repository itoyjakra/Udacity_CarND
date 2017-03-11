import numpy as np
import cv2
from pprint import pprint as pp
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Lane(object):
    """docstring for Frame."""
    def __init__(self, image, warped, window_params):
        w_width, w_height, w_margin = window_params
        self.image = image
        self.warped_image = warped
        self.window_width = w_width
        self.window_height = w_height
        self.margin = w_margin

    def window_mask(self, center, level):
        """
        Returns a rectangular mask around a detected lane center
        """
        y_start = int(self.warped_image.shape[0]-(level+1)*self.window_height)
        y_end = int(self.warped_image.shape[0]-level*self.window_height)
        x_start = max(0,int(center-self.window_width/2))
        x_end = min(int(center+self.window_width/2),self.warped_image.shape[1])

        output = np.zeros_like(self.warped_image)
        output[y_start:y_end, x_start:x_end] = 1

        return output

    def find_window_centroids(self):
        """
        Find the centroids at each level (horizontal slice) of the left and right lanes
        """
        window_centroids = [] # Store the (left,right) window centroid positions per level
        window = np.ones(self.window_width) # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(self.warped_image[int(3*self.warped_image.shape[0]/4):,:int(self.warped_image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-self.window_width/2
        r_sum = np.sum(self.warped_image[int(3*self.warped_image.shape[0]/4):,int(self.warped_image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-self.window_width/2+int(self.warped_image.shape[1]/2)

        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(self.warped_image.shape[0]/self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(self.warped_image[int(self.warped_image.shape[0]-(level+1)*self.window_height):int(self.warped_image.shape[0]-level*self.window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)

            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width/2
            l_min_index = int(max(l_center+offset-self.margin,0))
            l_max_index = int(min(l_center+offset+self.margin,self.warped_image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-self.margin,0))
            r_max_index = int(min(r_center+offset+self.margin,self.warped_image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

            # Add what we found for that layer
            window_centroids.append((l_center,r_center))

        return window_centroids

    def fit_poly_to_line(self, window_centroids, scale=None):
        """
        Fit polynomial to the two detected lane markings
        """
        if scale is None:
            leftx = np.array(window_centroids)[:,0]
            rightx = np.array(window_centroids)[:,1]
            y = np.arange(len(window_centroids), 0, -1)*self.window_height - self.window_height/2
        else:
            scalex, scaley = scale
            leftx = np.array(window_centroids)[:,0] * scalex
            rightx = np.array(window_centroids)[:,1] * scalex
            y = np.arange(len(window_centroids), 0, -1)*self.window_height - self.window_height/2
            y *= scaley

        left_fit = np.polyfit(y, leftx, 2)
        right_fit = np.polyfit(y, rightx, 2)

        if scale is None:
            return (left_fit, right_fit)
        else:
            return (left_fit, right_fit, y)

    def plot_lane(self, Minv, lane_par, window_centroids=None, plotfig=False):
        """
        highlight the detected lane and overlay it
        on the original image
        """
        roc, offset = lane_par
        if offset < 0:
            side = 'right'
        else:
            side = 'left'
        if window_centroids==None:
            window_centroids = self.find_window_centroids()

        left_fit, right_fit = self.fit_poly_to_line(window_centroids)
        n = self.image.shape[0]
        ploty = np.linspace(0, n-1, num=n)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        left_fit, right_fit = self.fit_poly_to_line(window_centroids)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (self.image.shape[1], self.image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self.image, 1, newwarp, 0.3, 0)

        if plotfig:
            plt.figure()
            plt.imshow(result)
            plt.text(30, 50, "Radius of Curvature = %f m" % roc)
            plt.text(30, 100, "Vehicle is %s m %s of the center" % (np.abs(offset), side))
            plt.show()

        return result, roc, offset

    def radius_of_curvature(self, centroids):
        """
        Calculate the radius of curvature for the two lane markings
        """
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit_cr, right_fit_cr, ploty = self.fit_poly_to_line(centroids, scale=[xm_per_pix, ym_per_pix])

        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        avg_roc = (left_curverad + right_curverad)/2
        lx = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
        rx = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
        lane_center = lx + (rx - lx)/2
        image_center = self.image.shape[1]/2*xm_per_pix
        vehicle_offset = image_center - lane_center

        return (avg_roc, vehicle_offset)

    def display_lane_centers(self, window_centroids=None):
        """
        plot window centroids for each lane and the mask around it
        """
        if window_centroids==None:
            window_centroids = self.find_window_centroids()

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(self.warped_image)
            r_points = np.zeros_like(self.warped_image)

            # Go through each level and draw the windows
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(window_centroids[level][0],level)
                r_mask = self.window_mask(window_centroids[level][1],level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channle
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            warpage = np.array(cv2.merge((self.warped_image,self.warped_image,self.warped_image)),np.uint8) # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((self.warped_image,self.warped_image,self.warped_image)),np.uint8)

        # Display the final results

        y = np.arange(len(window_centroids), 0, -1)*self.window_height - self.window_height/2
        left_points = np.array([(x, y) for (x, y) in zip(np.array(window_centroids)[:,0], y)])
        right_points = np.array([(x, y) for (x, y) in zip(np.array(window_centroids)[:,1], y)])

        left_fit, right_fit = self.fit_poly_to_line(window_centroids)
        n = self.image.shape[0]
        ploty = np.linspace(0, n-1, num=n)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        plt.plot()
        plt.imshow(output)
        plt.plot (left_points[:,0], left_points[:,1], 'ro')
        plt.plot (right_points[:,0], right_points[:,1], 'bo')
        plt.plot (left_fitx, ploty, '-c')
        plt.plot (right_fitx, ploty, '-m')
        plt.title('window fitting results')
        plt.show()
