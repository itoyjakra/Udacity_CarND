import matplotlib.pyplot as plt
from Frame import *

def plot_image(im, cmap=None):
    if cmap is None:
        plt.imshow(im)
    else:
        plt.imshow(im, cmap=cmap)
    plt.show()

def main():
  cal = CalibrateCamera()
  f = cv2.imread('test_images/signs_vehicles_xygrad.png')

  im = Frame(f, cal)
  print (im.image.shape)
  im.mag_thresh(sobel_kernel=15, mag_thresh=(90, 125))
  #im.mag_thresh()
  plot_image(im.grad_mag_filtered, cmap='gray')

if __name__ == '__main__':
    main()
