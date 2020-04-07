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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(grad_mag)/255
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(grad_mag)
    mag_binary[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(abs_grad_dir)
    dir_binary[(abs_grad_dir) >= thresh[0] & (abs_grad_dir <= thresh[1])] = 1

    return dir_binary

def hls_thresh(img, thresh=(100, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    hls_binary = np.zeros_like(s_channel)
    hls_binary[(s_channel)>thresh[0] & (s_channel <= thresh[1])] = 1
    return hls_binary

def combine_thresh(image, ksize):
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(50, 255))
    #grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(0, 255))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
    hls_binary = hls_thresh(image, thresh=(170, 255))

    combine_out = np.zeros_like(dir_binary)
    combine_out[(gradx == 1 | ((mag_binary == 1) & (dir_binary == 1))) | hls_binary == 1 ]
    return combine_out, gradx, mag_binary, dir_binary, hls_binary

if __name__ == '__main__':
    img = mpimg.imread('test_images/straight_lines1.jpg') 
    # mtx = [[1.15715431e+03, 0.00000000e+00, 6.64959703e+02],
    #    [0.00000000e+00, 1.15206977e+03, 3.86966325e+02],
    #    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    # dist = [-0.2379688 , -0.09158468, -0.00114565,  0.00025177,  0.13698797]  
    with open('camera_cali_parameters.pickle', 'rb') as f:
        para = pickle.load(f)
    mtx = para['mtx']
    dist = para['dist']
    print(mtx)
    print(dist)

    img = cv2.undistort(img, mtx, dist, None, None)
    combine_out, gradx, mag_binary, dir_binary, hls_binary = combine_thresh(img,3)

    outout = [combine_out, gradx, mag_binary, dir_binary, hls_binary]
    for i in range(5):
        print(i)
        plt.subplot(2, 3, i+1)
        plt.imshow(outout[i], cmap='gray')
    plt.show()
    plt.savefig('example_images/thresh_im.jpg')

