import matplotlib.pyplot as plt
import numpy as np
import cv2
from pprint import pprint as pp
import glob
import matplotlib.image as mpimg
from Frame import Frame
from Lane import Lane
from moviepy.editor import VideoFileClip, ImageSequenceClip

def plot_image(im, cmap=None):
    plt.figure()
    if cmap is None:
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=cmap)
    plt.show()

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

    return (M, dst, src)

def one_frame_pipeline(image, params, plotfig=False):
    """
    Pipeline for processing an individual image
    """
    if type(image) is str:
        f = mpimg.imread(image)
    elif type(image is np.ndarray):
        f = np.copy(image)
    else:
        print ("unknown image type")
        return None

    mtx, dist, dst, src, M, Minv = params
    undist = cv2.undistort(f, mtx, dist, None, mtx)
    im = Frame(undist)

    # apply color channel thresholds
    ch1 = im.hsv_select(thresh=(150, 255), channel=1)
    ch2 = im.hsv_select(thresh=(200, 255), channel=2)
    ch3 = im.rgb_select(thresh=(0, 30))

    # apply Sobel gradient thresholds
    dir_binary = im.dir_thresh(sobel_kernel=15, thresh=(0.5, 1.1))
    mag_binary = im.mag_thresh(sobel_kernel=15, thresh=(20, 125))
    sobel_binary = im.sobel_thresh(sobel_kernel=15, orient='x', thresh=(200, 255))

    # combine filters
    combined = np.zeros_like(dir_binary)
    combined[(mag_binary == 1) & (dir_binary == 1) | (sobel_binary == 1)] = 1
    combined = combined.astype(np.uint8)
    ch = ((ch1 | ch2) & ch3) | combined

    if plotfig:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[24, 6])
        ax1.imshow(f)
        ax2.imshow(ch, cmap='gray')
        plt.show()

    # warp image for lane detection
    img_size = (f.shape[1], f.shape[0])
    warped = cv2.warpPerspective(ch, M, img_size)

    if plotfig:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[24, 6])
        ax1.imshow(ch, cmap='gray')
        ax2.imshow(warped, cmap='gray')
        plt.show()

    # scan warped image for lanes
    window_width = 50
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    window_params = (window_width, window_height, margin)
    lane = Lane(undist, warped, window_params)
    cents = lane.find_window_centroids()
    roc, offset = lane.radius_of_curvature(cents)
    if plotfig:
        lane.display_lane_centers(cents)
    return lane.plot_lane(Minv, (roc, offset), window_centroids=cents)

def video_pipeline(video_file, params):
    video_clip = VideoFileClip(video_file)
    image_sequence = []
    for i, f, in enumerate(video_clip.iter_frames()):
        print ('processing frame ', i)
        img, roc, offset = one_frame_pipeline(f, params)
        if offset < 0:
            side = 'right'
        else:
            side = 'left'
        cv2.putText(img, "Radius of Curvature = %.1f m" % roc, (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))
        cv2.putText(img, "Vehicle is %.2f m %s of the center" % (np.abs(offset), side), (70, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255))
        image_sequence.append(img)
        clip = ImageSequenceClip(image_sequence, fps=video_clip.fps)
        clip.write_videofile("test_images/challenge_test.mp4", audio=False)
        if i>10:
            break

def main():
    # TODO
    # 1. undistort the image at the
    # 2. in find_window_centroids generalize the bottom quarter selection
    mtx, dist = CalibrateCamera()
    M, dst, src = PerspectiveTransform(plotfig=False)
    Minv = cv2.getPerspectiveTransform(dst, src)
    params = (mtx, dist, dst, src, M, Minv)

    video_pipeline("test_images/project_video.mp4", params)
    assert 3==4
    test_files = glob.glob('test_images/*.jpg')
    one_frame_pipeline("test_images/straight_lines1.jpg", params, plotfig=False)

    for f in test_files:
        print ('processing ', f)
        one_frame_pipeline(f, params, plotfig=False)

if __name__ == '__main__':
    main()
