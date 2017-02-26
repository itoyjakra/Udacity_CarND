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
from scipy.ndimage.measurements import label

def read_images(dataset):
    # Read in cars and notcars
    if dataset == 'small':
        cars = glob.glob('smallset/vehicles_smallset/**/*.jpeg', recursive=True)
        print ('number of cars = ', len(cars))
        notcars = glob.glob('smallset/non-vehicles_smallset/**/*.jpeg', recursive=True)
        print ('number of non-cars = ', len(notcars))
        print ('image size for cars: ', mpimg.imread(cars[0]).shape)
        print ('image size for non-cars: ', mpimg.imread(notcars[0]).shape)
        image_shape = mpimg.imread(cars[0]).shape[:2]
        image_type = 'jpeg'
    elif dataset == 'GTI':
        cars = glob.glob('vehicles/GTI*/*.png', recursive=True)
        print ('number of cars = ', len(cars))
        notcars = glob.glob('non-vehicles/GTI*/*.png', recursive=True)
        print ('number of non-cars = ', len(notcars))
        print ('image size for cars: ', mpimg.imread(cars[0]).shape)
        print ('image size for non-cars: ', mpimg.imread(notcars[0]).shape)
        image_shape = mpimg.imread(cars[0]).shape[:2]
        image_type = 'png'
    elif dataset == 'all':
        cars = glob.glob('vehicles/**/*.png', recursive=True)
        print ('number of cars = ', len(cars))
        notcars = glob.glob('non-vehicles/**/*.png', recursive=True)
        print ('number of non-cars = ', len(notcars))
        print ('image size for cars: ', mpimg.imread(cars[0]).shape)
        print ('image size for non-cars: ', mpimg.imread(notcars[0]).shape)
        image_shape = mpimg.imread(cars[0]).shape[:2]
        image_type = 'png'
    else:
        print ('invalid data set')
        return None

    return (cars, notcars, image_type, image_shape)

def sample(cars, notcars):
    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    return (cars, notcars)

def train_model(dataset):
    '''
    train a model on the given dataset based on the
    best parameters and return the model
    '''

    # read the images
    imgs = read_images(dataset)
    training_image_type, training_image_shape = imgs[2:]

    # set parameters
    params = {'color_space': 'YCrCb',
                  'orient': 9,
                  'pix_per_cell': 8,
                  'cell_per_block': 2,
                  'hog_channel': "ALL",
                  'spatial_size': (32, 32),
                  'hist_bins': 10,
                  'spatial_feat': True,
                  'hist_feat': False,
                  'hog_feat': True,
                  'y_start_stop': [None, None],
                  'training_image_shape': training_image_shape,
                  'training_image_type': training_image_type
                 }
    #param_print_list = []
    #for key, value in param_grid.items():
    #    if len(value) > 1:
    #        param_print_list.append(key)

    cfeat, ncfeat = runme_p(imgs, params)
    features = (cfeat, ncfeat)
    clf, scaler, acc = create_model(features)

    print ('Accuracy = ', acc)
    print ('__________________________________________')

    return (clf, scaler, params)

def param_search(dataset):

    # read the images
    imgs = read_images(dataset)

    # sample if needed
    #cars, notcars = sample(cars, notcars)

    '''
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
    '''

    param_grid = {'color_space': ['YCrCb'],
                  'orient': [9],
                  'pix_per_cell': [8],
                  'cell_per_block': [2],
                  'hog_channel': ["ALL"],
                  'spatial_size': [(32, 32)],
                  'hist_bins': [10],
                  'spatial_feat': [True],
                  'hist_feat': [False],
                  'hog_feat': [True],
                  'y_start_stop': [[None, None]]
                 }
    param_print_list = []
    for key, value in param_grid.items():
        if len(value) > 1:
            param_print_list.append(key)

    print ('parameters used ...')
    for key, val in param_grid.items():
        print (key, ':', val)

    if len(param_print_list) > 0:
        print ('parameters used in grid search ...')
        print (param_print_list)

    for i, params in enumerate(list(ParameterGrid(param_grid))):
        cfeat, ncfeat = runme_p(imgs, params)
        features = (cfeat, ncfeat)
        clf, scaler, acc = create_model(features)
        imtype, imshape = imgs[2:]
        plot_bounding_box_old(clf, scaler, params, i, imtype, imshape)

        print ('________________%d________________________' %i)
        for key, value in params.items():
            if key in param_print_list:
                print (key, params[key])
        print ('Accuracy = ', acc)
        print ('__________________________________________')

def runme_p(imgs, params):

    # get image info
    cars, notcars, image_type, image_shape = imgs

    # extract parameters from dictionary
    color_space = params['color_space']
    spatial_size = params['spatial_size']
    hist_bins = params['hist_bins']
    orient = params['orient']
    pix_per_cell = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel = params['hog_channel']
    spatial_feat = params['spatial_feat']
    hist_feat = params['hist_feat']
    hog_feat = params['hog_feat']

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

    return (car_features, notcar_features)


def create_model(features):
    car_features, notcar_features = features
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

    #print('Using:',orient,'orientations',pix_per_cell,
    #    'pixels per cell and', cell_per_block,'cells per block')
    #print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    accuracy = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', accuracy)
    #print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    return (svc, X_scaler, accuracy)

def plot_bounding_box(image_file, clf, scaler, params):
    '''
    given a new image, plot a bounding box
    around all the cars detected by the
    classification algorithm.
    '''
    image = mpimg.imread(image_file)
    draw_image = np.copy(image)
    imy, imx = image.shape[:2]
    image_type = image_file.split('.')[-1]

    if (image_type=='jpeg' or image_type=='jpg') and params['training_image_type']=='png':
        image = image.astype(np.float32)/255

    # extract parameters from dictionary
    color_space = params['color_space']
    spatial_size = params['spatial_size']
    hist_bins = params['hist_bins']
    orient = params['orient']
    pix_per_cell = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel = params['hog_channel']
    spatial_feat = params['spatial_feat']
    hist_feat = params['hist_feat']
    hog_feat = params['hog_feat']
    training_image_shape = params['training_image_shape']

    # multi box search
    min_fraction = 6
    max_fraction = 10
    step_fraction = 2
    aspect_ratio = 1.0
    hot_windows = []
    start_scan = int(imy/2)
    for f in range(min_fraction, max_fraction+1, step_fraction):
        y_start_stop = [start_scan, min(start_scan + (f - 1) * int(imy/f), imy)] # Min and max in y to search in slide_window()
        xy_window = (int(imy/f), int(imy/f*aspect_ratio))
        windows = ro.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=xy_window, xy_overlap=(0.5, 0.5))
        print (f, len(windows))

        new_hot_windows = ro.search_windows(image, windows, clf, scaler, training_image_shape, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
        print (f, y_start_stop, xy_window, len(new_hot_windows)) #, new_hot_windows)
        if len(new_hot_windows) > 0:
            hot_windows = hot_windows + new_hot_windows

    window_img = ro.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=3)
    #plt.figure()
    #plt.imshow(window_img)
    #plt.show()

    heatmap = np.zeros([draw_image.shape[0], draw_image.shape[1]])
    heatmap = ro.add_heat(heatmap, hot_windows)

    heatmap = ro.apply_threshold(heatmap, 2)
    labels = label(heatmap)
    print(labels[1], 'cars found')

    draw_img = ro.draw_labeled_bboxes(np.copy(draw_image), labels)

    return draw_img

def plot_bounding_box_old(clf, scaler, params, index, image_type, image_shape):
    '''
    given a new image, plot a bounding box
    around all the cars detected by the
    classification algorithm.
    '''
    print ('image_shape = ', image_shape)
    image = mpimg.imread('bbox-example-image.jpg')
    image = mpimg.imread('test/frame300.jpg')
    draw_image = np.copy(image)
    imy, imx = image.shape[:2]
    print ('in old: ', image_type, image.shape)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    if image_type == 'png':
        image = image.astype(np.float32)/255

    # extract parameters from dictionary
    color_space = params['color_space']
    spatial_size = params['spatial_size']
    hist_bins = params['hist_bins']
    orient = params['orient']
    pix_per_cell = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel = params['hog_channel']
    spatial_feat = params['spatial_feat']
    hist_feat = params['hist_feat']
    hog_feat = params['hog_feat']

    # multi box search
    min_fraction = 6
    max_fraction = 10
    step_fraction = 2
    aspect_ratio = 1.0
    hot_windows = []
    start_scan = int(imy/2)
    for f in range(min_fraction, max_fraction+1, step_fraction):
        y_start_stop = [start_scan, min(start_scan + (f - 1) * int(imy/f), imy)] # Min and max in y to search in slide_window()
        xy_window = (int(imy/f), int(imy/f*aspect_ratio))
        windows = ro.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=xy_window, xy_overlap=(0.5, 0.5))
        print (f, len(windows))

        new_hot_windows = ro.search_windows(image, windows, clf, scaler, image_shape, color_space=color_space,
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
    '''
    plt.show()
    '''
    plt.imshow(window_img)
    fig1 = plt.gcf()
    plt.draw()
    fig1.savefig('new_fig%d.jpg' %index, dpi=100)


    heatmap = np.zeros([draw_image.shape[0], draw_image.shape[1]])
    heatmap = ro.add_heat(heatmap, hot_windows)
    plt.figure()
    plt.imshow(heatmap, cmap='gray')


    heatmap = ro.apply_threshold(heatmap, 1)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    plt.figure()
    plt.imshow(labels[0], cmap='gray')

    draw_img = ro.draw_labeled_bboxes(np.copy(draw_image), labels)
    # Display the image
    plt.figure()
    plt.imshow(draw_img)

    plt.show()

def get_still_from_video(files_to_glob):
    #files = glob.glob('test/frame3[0-3]?.jpg')
    files = glob.glob(files_to_glob)
    for file in files:
        yield file

def process_video(files, clf, scaler, params):
    for filename in get_still_from_video(files):
        print ('processing file:', filename)
        fig = plot_bounding_box(filename, clf, scaler, params)
        plt.figure()
        plt.imshow(fig)
        plt.show()

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

    cars, notcars, image_type, image_shape = read_images()
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
    #param_search('all')

    clf, scaler, params = train_model('all')
    '''
    new_image_file='test/frame300.jpg'
    fig = plot_bounding_box(new_image_file, clf, scaler, params)
    plt.figure()
    plt.imshow(fig)
    plt.show()
    '''
    glob_files = 'test/frame3[0-3]?.jpg'
    process_video(glob_files, clf, scaler, params)
