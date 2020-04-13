import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

print('start...')
def camera_calibration():
# Camera Calibration
    nx = 9
    ny = 6
    points = []
    corners = []

    for i in range(1,20):
    
        p = np.zeros((nx*ny,3), np.float32)
        p[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)	
    
        img = cv2.imread('camera_cal/calibration%s.jpg'%str(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corner = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            points.append(p)
            corners.append(corner)
        else:
            print('image'+'%s'%str(i)+' has problem')
    img = cv2.imread('test_images/straight_lines1.jpg')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, _, _ = cv2.calibrateCamera(points, corners, img_size, None, None)
    return mtx, dist



if __name__ == "__main__":
    mtx, dist = camera_calibration()
    coefficient = {'mtx': mtx,'dist': dist}
    print(coefficient) 
    with open('camera_cali_parameters.pickle', 'wb') as f:
        pickle.dump(coefficient, f)
        
    # For testing
    img = mpimg.imread('test_images/straight_lines1.jpg')
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    plt.imshow(dst)
    plt.savefig('example_images/undistorted_im.jpg')
