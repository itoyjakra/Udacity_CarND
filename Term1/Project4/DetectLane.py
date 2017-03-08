import matplotlib.pyplot as plt
from Frame import *

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
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[24, 6])

    im = Frame(f)
    ch1 = im.hsv_select(thresh=(150, 255), channel=1)
    ch2 = im.hsv_select(thresh=(200, 255), channel=2)
    ch3 = im.rgb_select(thresh=(0, 30))

    #sobel = im.sobel(sobel_kernel=15)
    dir_binary = im.dir_thresh(sobel_kernel=15, thresh=(0.5, 1.1))
    mag_binary = im.mag_thresh(sobel_kernel=15, thresh=(20, 125))
    sobel_binary = im.sobel_thresh(sobel_kernel=15, orient='x', thresh=(200, 255))

    combined = np.zeros_like(dir_binary)
    combined[(mag_binary == 1) & (dir_binary == 1) | (sobel_binary == 1)] = 1
    combined = combined.astype(np.uint8)
    ch = ((ch1 | ch2) & ch3) | combined

    if plotfig:
        ax1.imshow(f)
        ax2.imshow(ch, cmap='gray')
        plt.show()

def main():
    one_frame_pipeline('test_images/test6.jpg', plotfig=True)

if __name__ == '__main__':
    main()
