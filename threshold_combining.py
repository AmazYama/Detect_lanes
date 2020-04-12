from cv2 import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0,255)):
    print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(grad_mag)
    mag_binary[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(abs_grad_dir)
    dir_binary[(abs_grad_dir >= thresh[0]) & (abs_grad_dir <= thresh[1])] = 1

    return dir_binary

def hls_thresh(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    hls_binary = np.zeros_like(s_channel)
    hls_binary[(s_channel>thresh[0]) & (s_channel <= thresh[1])] = 1
    return hls_binary

def combine_thresh(image, ksize):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(70, 255))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(100, 255))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(100, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize*5, thresh=(0.8, 1.2))
    hls_binary = hls_thresh(image, thresh=(200, 255))
    combine_out = np.zeros_like(dir_binary)
    combine_out[((mag_binary ==1) & (dir_binary ==1)| (grady==1) & (gradx == 1) ) | (hls_binary == 1)]=1
    return combine_out, gradx, grady, mag_binary, dir_binary, hls_binary

if __name__ == '__main__':
    img = mpimg.imread('test_images/straight_lines1.jpg') 
    with open('camera_cali_parameters.pickle', 'rb') as f:
        para = pickle.load(f)
    mtx = para['mtx']
    dist = para['dist']
    print(mtx)
    print(dist)

    img = cv2.undistort(img, mtx, dist, None, None)
    combine_out, gradx, grady, mag_binary, dir_binary, hls_binary = combine_thresh(img,3)

    outout = [combine_out, gradx, grady, mag_binary, dir_binary, hls_binary]
    for i in range(6):
        print(i)
        plt.subplot(2, 3, i+1)
        plt.imshow(outout[i], cmap='gray')
    plt.savefig('example_images/thresh_im.jpg')

