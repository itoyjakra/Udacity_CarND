import time
import Routines as ro
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import ParameterGrid
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def read_images():
    # Read in cars and notcars
    cars = glob.glob('smallset/vehicles_smallset/**/*.jpeg', recursive=True)
    print ('number of cars = ', len(cars))
    notcars = glob.glob('smallset/non-vehicles_smallset/**/*.jpeg', recursive=True)
    print ('number of non-cars = ', len(notcars))
    print ('image size for cars: ', mpimg.imread(cars[0]).shape)
    print ('image size for non-cars: ', mpimg.imread(notcars[0]).shape)
    image_shape = mpimg.imread(cars[0]).shape[:2]

    return (cars, notcars, image_shape)

def sample(cars, notcars):
    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    return (cars, notcars)

def runme():
    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (12, 12) # Spatial binning dimensions
    hist_bins = 20    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()

    cars, notcars, image_shape = read_images()
    #cars, notcars = sample(cars, notcars)

    car_features = ro.extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = ro.extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    image = mpimg.imread('bbox-example-image.jpg')
    draw_image = np.copy(image)
    print ('shape of test image: ', image.shape)
    imy, imx = image.shape[:2]

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255

    # multi box search
    min_fraction = 6
    max_fraction = 6
    aspect_ratio = 1.0
    hot_windows = []
    start_scan = int(imy/2)
    for f in range(min_fraction, max_fraction+1):
        y_start_stop = [start_scan, min(start_scan + (f - 1) * int(imy/f), imy)] # Min and max in y to search in slide_window()
        xy_window = (int(imy/f), int(imy/f*aspect_ratio))
        windows = ro.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=xy_window, xy_overlap=(0.5, 0.5))
        print (f, len(windows))

        new_hot_windows = ro.search_windows(image, windows, svc, X_scaler, image_shape, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)                       
        print (f, y_start_stop, xy_window, len(new_hot_windows)) #, new_hot_windows)
        if len(new_hot_windows) > 0:
            hot_windows = hot_windows + new_hot_windows

    print ('size of hot_windows before plotting: ', len(hot_windows))
    window_img = ro.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=3)

    plt.imshow(window_img)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('tessstttyyy.jpg', dpi=100)

if __name__=='__main__':
    runme()
