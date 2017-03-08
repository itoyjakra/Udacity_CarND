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
    ch = (ch1 | ch2) & ch3

    if plotfig:
        ax1.imshow(f)
        ax2.imshow(ch1, cmap='gray')
        plt.show()

def main():
    one_frame_pipeline('test_images/test6.jpg', plotfig=True)

if __name__ == '__main__':
    main()
