from Lane import Lane
import numpy as np
import cv2
import matplotlib.pyplot as plt

class LaneSeries(Lane):
    """docstring for Frame."""
    def __init__(self, image, warped):
        self.image = image          # original image
        self.warped_image = warped  # warped image
        self.frame_counter = 0      # frame number
        # convolution window parameters
        self.window_width = 50
        self.window_height = 80
        # peak values of convoluted signals
        self.left_maxval = -100
        self.right_maxval = -100
        # length of running list of lane attributes
        self.len_mov_av = 6
        # how many detection failures to allow before fresh lane search
        self.fail_count_tolerance = 3
        # maintain a running list of frame numbers where lane detection failed
        self.left_failed_frames = []
        self.right_failed_frames = []
        # maintain a running list of lane centroids
        self.left_centroid_list = []
        self.right_centroid_list = []
        # maintain a running list of lane radius of curvatures
        self.left_roc_list = []
        self.right_roc_list = []
        # accept centroids away from car only if it's convolution peak is a good fraction
        # of that of the nearest centroid
        self.intens_accep_frac = 10
        self.lane_shift_allowance = 50
        # acceptable RMSE between previously detected centroids and the new ones
        self.max_lane_shift_rmse = 25
        # roc test skipped when lanes are very straight
        self.min_roc_threshold = 8000

        self.window_centroids = []
        self.roc = 0
        self.vehicle_offset = 0
        self.lane_on_image = None
        # search margin for new lane search
        self.initial_margin = 40
        # search margin for lanes based on previous detection
        self.margin = np.linspace(20, 30, int(self.image.shape[0]/self.window_height))

    def init_lane_centers(self):
        """
        initialize lane centers based on bottom fourth of the frame
        """
        window = np.ones(self.window_width)
        # Sum quarter bottom of image to get slice
        l_sum = np.sum(self.warped_image[int(3*self.warped_image.shape[0]/4):,:int(self.warped_image.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-self.window_width/2
        r_sum = np.sum(self.warped_image[int(3*self.warped_image.shape[0]/4):,int(self.warped_image.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - self.window_width/2+int(self.warped_image.shape[1]/2)
        l_maxval = np.max(np.convolve(window, l_sum))
        r_maxval = np.max(np.convolve(window, r_sum))

        return (l_center, r_center, l_maxval, r_maxval)

    def find_window_centroids_this_frame(self):
        """
        Find the centroids at each level (horizontal slice) of the left and right lanes
        by searching areas near previous centroids
        """
        l_centers = []
        r_centers = []
        window = np.ones(self.window_width)
        n_levels = (int)(self.warped_image.shape[0]/self.window_height)

        for level in range(n_levels):
            v_start = int(self.warped_image.shape[0] - (level+1)*self.window_height)
            v_end = int(self.warped_image.shape[0] - level*self.window_height)
            image_layer = np.sum(self.warped_image[v_start:v_end,:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            offset = self.window_width/2

            scan_center_left, scan_center_right = self.window_centroids[level]

            l_min_index = int(max(scan_center_left + offset - self.margin[level], 0))
            l_max_index = int(min(scan_center_left + offset + self.margin[level], self.warped_image.shape[1]))
            signal = conv_signal[l_min_index:l_max_index]
            if len(signal) > 0:
                l_centers.append(np.argmax(signal) + l_min_index - offset)
            else:
                l_centers.append(self.window_centroids[level, 0])
            #l_centers.append(np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset)

            r_min_index = int(max(scan_center_right + offset - self.margin[level], 0))
            r_max_index = int(min(scan_center_right + offset + self.margin[level], self.warped_image.shape[1]))
            signal = conv_signal[r_min_index:r_max_index]
            if len(signal) > 0:
                r_centers.append(np.argmax(signal) + r_min_index - offset)
            else:
                r_centers.append(self.window_centroids[level, 1])
            #r_centers.append(np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index-offset)

        y = self.image.shape[0] - np.arange(n_levels) * (0.5 + self.window_height)
        left_fit = np.polyfit(y, np.array(l_centers), 2)
        right_fit = np.polyfit(y, np.array(r_centers), 2)

        eval_y = y
        lx = left_fit[0]*eval_y**2 + left_fit[1]*eval_y + left_fit[2]
        rx = right_fit[0]*eval_y**2 + right_fit[1]*eval_y + right_fit[2]

        window_centroids_this_frame = np.array([(l, r) for l, r in zip(lx, rx)])

        return window_centroids_this_frame, left_fit, right_fit

    def find_window_centroids_first_frame(self):
        """
        Find the centroids at each level (horizontal slice) of the left and right lanes
        for the first time
        """
        window_centroids = []
        stripes = []
        weights = []

        window = np.ones(self.window_width)
        l_center, r_center, l_maxval, r_maxval = self.init_lane_centers()
        n_levels = (int)(self.warped_image.shape[0]/self.window_height)
        self.left_maxval = l_maxval
        self.right_maxval = r_maxval

        if (len(self.window_centroids) > 0) and (np.abs(l_center - self.window_centroids[0, 0]) > self.margin[0]):
            #l_center = self.window_centroids[0, 0]
            print ("using previous value for l_center")
        if (len(self.window_centroids) > 0) and (np.abs(r_center - self.window_centroids[0, 1]) > self.margin[0]):
            #r_center = self.window_centroids[0, 1]
            print ("using previous value for r_center")

        window_centroids.append((l_center, r_center))
        stripes.append(0)
        weights.append((l_maxval, r_maxval))

        for level in range(1, n_levels):
            v_start = int(self.warped_image.shape[0] - (level+1)*self.window_height)
            v_end = int(self.warped_image.shape[0] - level*self.window_height)
            image_layer = np.sum(self.warped_image[v_start:v_end,:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            offset = self.window_width/2

            l_min_index = int(max(l_center+offset-self.initial_margin,0))
            l_max_index = int(min(l_center+offset+self.initial_margin,self.warped_image.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center+offset-self.initial_margin,0))
            r_max_index = int(min(r_center+offset+self.initial_margin,self.warped_image.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            r_accept = np.max(conv_signal[r_min_index:r_max_index]) > r_maxval/self.intens_accep_frac
            l_accept = np.max(conv_signal[l_min_index:l_max_index]) > l_maxval/self.intens_accep_frac

            if r_accept and l_accept:
                window_centroids.append((l_center,r_center))
                weights.append((np.max(conv_signal[l_min_index:l_max_index]), np.max(conv_signal[r_min_index:r_max_index])))
                stripes.append(level)

        weights = np.array(weights)

        y = [ self.image.shape[0] - (s+0.5) * self.window_height for s in stripes]
        left_fit = np.polyfit(y, np.array(window_centroids)[:,0], 2, w=weights[:,0])
        right_fit = np.polyfit(y, np.array(window_centroids)[:,1], 2, w=weights[:,1])

        eval_y = self.image.shape[0] - np.arange(int(self.image.shape[0]/self.window_height)) * (0.5 + self.window_height)
        lx = left_fit[0]*eval_y**2 + left_fit[1]*eval_y + left_fit[2]
        rx = right_fit[0]*eval_y**2 + right_fit[1]*eval_y + right_fit[2]

        self.window_centroids = np.array([(l, r) for l, r in zip(lx, rx)])
        self.left_centroid_list.append(self.window_centroids[:,0])
        self.right_centroid_list.append(self.window_centroids[:,1])

        sanity = self.window_centroids[:,1] - self.window_centroids[:,0]
        assert (sanity > 0).all()
        #self.display_lane_centers()
        #self.margin = np.linspace(10, 120, int(self.image.shape[0]/self.window_height))



    def fit_poly_to_line(self, scale=None):
        """
        Fit polynomial to the two detected lane markings
        """
        if scale is None:
            leftx = np.array(self.window_centroids)[:,0]
            rightx = np.array(self.window_centroids)[:,1]
            y = np.arange(len(self.window_centroids), 0, -1)*self.window_height - self.window_height/2
        else:
            scalex, scaley = scale
            leftx = np.array(self.window_centroids)[:,0] * scalex
            rightx = np.array(self.window_centroids)[:,1] * scalex
            y_prep = np.arange(len(self.window_centroids), 0, -1)*self.window_height - self.window_height/2
            y = [item * scaley for item in y_prep]

        left_fit = np.polyfit(y, leftx, 2)
        right_fit = np.polyfit(y, rightx, 2)

        if scale is None:
            return (left_fit, right_fit)
        else:
            return (left_fit, right_fit, y)

    def radius_of_curvature(self):
        """
        Calculate the radius of curvature for the two lane markings
        """
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit_cr, right_fit_cr, ploty = self.fit_poly_to_line(scale=[xm_per_pix, ym_per_pix])

        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        avg_roc = (left_curverad + right_curverad)/2

        print ('frame # = %d' % self.frame_counter)
        lc = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
        rc = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])
        flag = False

        lx = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
        rx = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
        lane_center = lx + (rx - lx)/2
        image_center = self.image.shape[1]/2*xm_per_pix
        vehicle_offset = image_center - lane_center

        self.roc = avg_roc
        self.vehicle_offset = vehicle_offset

    def display_lane_centers(self):
        """
        plot window centroids for each lane and the mask around it
        """
        # If we found any window centers
        if len(self.window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(self.warped_image)
            r_points = np.zeros_like(self.warped_image)

            # Go through each level and draw the windows
            for level in range(0,len(self.window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = self.window_mask(self.window_centroids[level][0],level)
                r_mask = self.window_mask(self.window_centroids[level][1],level)
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

        y = np.arange(len(self.window_centroids), 0, -1)*self.window_height - self.window_height/2
        left_points = np.array([(x, y) for (x, y) in zip(np.array(self.window_centroids)[:,0], y)])
        right_points = np.array([(x, y) for (x, y) in zip(np.array(self.window_centroids)[:,1], y)])

        left_fit, right_fit = self.fit_poly_to_line()
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
        plt.savefig('tmp.jpg')
        plt.close()
        #plt.show()

    def plot_lane(self, Minv, plotfig=False):
        """
        highlight the detected lane and overlay it
        on the original image
        """

        side = 'middle'

        left_fit, right_fit = self.fit_poly_to_line()
        n = self.image.shape[0]
        ploty = np.linspace(0, n-1, num=n)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        left_fit, right_fit = self.fit_poly_to_line()

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

        self.lane_on_image = result

    def add_frame(self, image, warped):
        """
        Get a new image frame
        """
        self.image = image
        self.frame_counter += 1
        self.warped_image = warped

    def check_new_centroids(self, cents):
        """
        calculates the rmse between the new lane centroids
        and the existing lane centroids
        """
        l_cents = cents[:,0]
        r_cents = cents[:,1]
        l_delta = np.sqrt(np.mean((l_cents - self.window_centroids[:,0])**2))
        r_delta = np.sqrt(np.mean((r_cents - self.window_centroids[:,1])**2))
        return (l_delta, r_delta)

    def check_curvature(self, cents):
        """
        check if there is a sign mismatch in the two lane curvatures
        """
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        leftx = np.array(cents)[:,0]
        rightx = np.array(cents)[:,1]
        y = np.arange(len(self.window_centroids), 0, -1)*self.window_height - self.window_height/2

        left_fit = np.polyfit(y, leftx, 2)
        right_fit = np.polyfit(y, rightx, 2)

        y_eval = np.min(y)
        left_roc1 = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_roc1 = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        # evaluate radius of curvature at the middle of the vertical axis
        y_eval = np.max(y) #(np.max(y) + np.min(y))/2
        left_roc2 = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / (2*left_fit[0])
        right_roc2 = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / (2*right_fit[0])

        if (np.abs(left_roc2) > self.min_roc_threshold) or \
                (np.abs(right_roc2) > self.min_roc_threshold) or \
                len(self.left_roc_list) == 0:
            return True, (left_roc2, right_roc2)
        else:
            accept = (np.sign(left_roc2) == np.sign(self.left_roc_list[-1])) and \
                 (np.sign(right_roc2) == np.sign(self.right_roc_list[-1])) and \
                 (np.sign(left_fit[0]) == np.sign(right_fit[0]))
            return accept, (left_roc2, right_roc2)

    def save_new_centroids(self, cents):
        """
        save the new centroids if acceptance creiteria are met
        """
        l_delta, r_delta = self.check_new_centroids(cents)
        curvature_match, rocs = self.check_curvature(cents)

        if (l_delta < self.max_lane_shift_rmse) and (curvature_match):
            if len(self.left_centroid_list) < self.len_mov_av:
                self.left_centroid_list.append(cents[:,0])
                self.left_roc_list.append(rocs[0])
            else:
                print ("replacing L")
                self.left_centroid_list.pop(0)
                self.left_centroid_list.append(cents[:,0])
                self.left_roc_list.pop(0)
                self.left_roc_list.append(rocs[0])
        else:
            # make sure failure happens in consecutive frames
            if (len(self.left_failed_frames) > 0) and ((self.frame_counter - self.left_failed_frames[-1]) == 1):
                self.left_failed_frames.append(self.frame_counter)
            else:
                self.left_failed_frames = [self.frame_counter]

            if len(self.left_centroid_list) < self.len_mov_av:
                self.left_centroid_list.append(self.window_centroids[:,0])
                self.left_roc_list.append(self.left_roc_list[-1])
            else:
                print ("replacing L")
                self.left_centroid_list.pop(0)
                self.left_centroid_list.append(self.window_centroids[:,0])
                self.left_roc_list.pop(0)
                self.left_roc_list.append(self.left_roc_list[-1])

        if (r_delta < self.max_lane_shift_rmse) and (curvature_match):
            if len(self.right_centroid_list) < self.len_mov_av:
                self.right_centroid_list.append(cents[:,1])
                self.right_roc_list.append(rocs[1])
            else:
                print ("replacing R")
                self.right_centroid_list.pop(0)
                self.right_centroid_list.append(cents[:,1])
                self.right_roc_list.pop(0)
                self.right_roc_list.append(rocs[1])
        else:
            # make sure failure happens in consecutive frames
            if (len(self.right_failed_frames) > 0) and ((self.frame_counter - self.right_failed_frames[-1]) == 1):
                self.right_failed_frames.append(self.frame_counter)
            else:
                self.right_failed_frames = [self.frame_counter]

            if len(self.right_centroid_list) < self.len_mov_av:
                self.right_centroid_list.append(self.window_centroids[:,1])
                self.right_roc_list.append(self.right_roc_list[-1])
            else:
                print ("replacing L")
                self.right_centroid_list.pop(0)
                self.right_centroid_list.append(self.window_centroids[:,1])
                self.right_roc_list.pop(0)
                self.right_roc_list.append(self.right_roc_list[-1])

        print ("left delta = %.2f, right delta = %.2f" % (l_delta, r_delta))
        print ("left fail = %d, right fail = %d" %(len(self.left_failed_frames), len(self.right_failed_frames)))
        print ("left roc list = ", len(self.left_roc_list))

        self.window_centroids[:,0] = np.mean(self.left_centroid_list, axis=0)
        self.window_centroids[:,1] = np.mean(self.right_centroid_list, axis=0)

    def process(self, plotfig=False):
        """
        executes a step of the lane detection pipeline
        """
        first_frame = (self.left_maxval < 0) and (self.right_maxval < 0)
        if first_frame:
            print ("-----finding lane centers for the first time-----")
            self.find_window_centroids_first_frame()
            _, rocs = self.check_curvature(self.window_centroids)
            self.left_roc_list.append(rocs[0])
            self.right_roc_list.append(rocs[1])
        else:
            print ("---moving onto frame---  ", self.frame_counter)
            new_centroids, left_fit, right_fit = self.find_window_centroids_this_frame()
            print ("new centroids: ", new_centroids)
            #fits = np.array([(l, r) for (l, r) in zip(left_fit, right_fit)])
            self.save_new_centroids(new_centroids)
            print ("lengths of cents: ", len(self.left_centroid_list), len(self.right_centroid_list))

        self.radius_of_curvature()
        self.display_lane_centers()
        if plotfig:
            self.display_lane_centers()

        if (len(self.left_failed_frames) >= self.fail_count_tolerance) or (len(self.right_failed_frames) >= self.fail_count_tolerance):
            return True
        else:
            return False

    def reset_lanes(self):
        """
        recalculate lanes following detection failure
        """
        print ("left failed frames: ", self.left_failed_frames)
        print ("right failed frames: ", self.right_failed_frames)
        print ("resetting saved lane info ----------------------")
        self.left_maxval = -100
        self.right_maxval = -100
        self.left_failed_frames = []
        self.right_failed_frames = []
        self.left_centroid_list = []
        self.right_centroid_list = []
        self.left_roc_list = [self.left_roc_list[-1]]
        self.right_roc_list = [self.right_roc_list[-1]]
        #self.window_centroids = []
