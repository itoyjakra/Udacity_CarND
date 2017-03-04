import matplotlib.pyplot as plt
from Frame import *

def plot_image(im):
    plt.imshow(im)
    plt.show()

def main():
  f = cv2.imread('camera_cal/calibration1.jpg')
  im = Frame(f)
  plot_image(im.undistorted())

if __name__ == '__main__':
    main()
