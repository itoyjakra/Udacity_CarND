import matplotlib.pyplot as plt
from Frame import *
from Lane import *

def plot_image(im, cmap=None):
    plt.figure()
    if cmap is None:
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=cmap)
    plt.show()

def one_frame_pipeline(image_file, plotfig=False):
    """
    Pipeline for processing an individual image
    """
    f = mpimg.imread(image_file)
    mtx, dist = CalibrateCamera()
    undist = cv2.undistort(f, mtx, dist, None, mtx)
    im = Frame(undist)

    ch1 = im.hsv_select(thresh=(150, 255), channel=1)
    ch2 = im.hsv_select(thresh=(200, 255), channel=2)
    ch3 = im.rgb_select(thresh=(0, 30))

    dir_binary = im.dir_thresh(sobel_kernel=15, thresh=(0.5, 1.1))
    mag_binary = im.mag_thresh(sobel_kernel=15, thresh=(20, 125))
    sobel_binary = im.sobel_thresh(sobel_kernel=15, orient='x', thresh=(200, 255))

    combined = np.zeros_like(dir_binary)
    combined[(mag_binary == 1) & (dir_binary == 1) | (sobel_binary == 1)] = 1
    combined = combined.astype(np.uint8)
    ch = ((ch1 | ch2) & ch3) # | combined

    if plotfig:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[24, 6])
        ax1.imshow(f)
        ax2.imshow(ch, cmap='gray')
        plt.show()

    ##
    ##----------------------------------------
    M = PerspectiveTransform(plotfig=True)
    #img = mpimg.imread("test_images/test2.jpg")
    mtx, dist = CalibrateCamera()
    undist = cv2.undistort(f, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    offset = 100
    img_size = (gray.shape[1], gray.shape[0])
    #warped = cv2.warpPerspective(undist, M, img_size)
    warped = cv2.warpPerspective(ch, M, img_size)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[24, 6])
    ax1.imshow(ch, cmap='gray')
    ax2.imshow(warped, cmap='gray')
    plt.show()

    window_width = 50
    window_height = 80 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    window_params = (window_width, window_height, margin)
    lane = Lane(undist, warped, window_params)
    cents = lane.find_window_centroids()
    print (cents)

def main():
    # TODO
    # 1. undistort the image at the
    # 2. in find_window_centroids generalize the bottom quarter selection
    one_frame_pipeline('test_images/test2.jpg', plotfig=True)

if __name__ == '__main__':
    main()
