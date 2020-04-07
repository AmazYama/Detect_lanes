import numpy as np 
from cv2 import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import pickle 
from threshold_combining import combine_thresh

def perspective_transform(img):
    im_sz = (img.shape[0],img.shape[1])   #720,1280
    src = np.float32(
		[[200, 720],
		[1100, 720],
		[595, 450],
		[685, 450]])
    dst = np.float32(
        [[300, 720],
		[980, 720],
		[300, 0],
		[980, 0]]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    #Minv = cv2.getPerspectiveTransform(dist, src)

    # Create warped image - use linear interpolation
    warped = cv2.warpPerspective(img, M, im_sz, flags=cv2.INTER_LINEAR)
    #unwarped = cv2.warpPerspective(warped, Minv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M

if __name__=='__main__':
    img = mpimg.imread('test_images/test5.jpg')
    with open('camera_cali_parameters.pickle','rb') as f:
        output = pickle.load(f)
    mtx = output['mtx']
    dist = output['dist']
    print(mtx)
    print(dist)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    combine_out, gradx, mag_binary, dir_binary, hls_binary = combine_thresh(img,3)
    warped, M = perspective_transform(img)
    plt.imshow(warped, cmap='gray')
    plt.show()
