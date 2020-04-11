# Lane Detection

### The goals / steps of this project are the following:
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## 1. Camera calibration
The code for this section refers to `camera_calibration.py`. I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `p` is just a replicated array of coordinates, and `points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `corners` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I then used the output `points` and `corners` to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result.

## 2, Pipeline (Single images)

1. To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:


2. Create binary image using color transform,, gradient, hsl transform etc. The detailed steps refer to the file `threshold_combining`.

3. The code for my perspective transform is includes a function called warper(), which is in the file `perspective_trans.py`. The warper() function takes as inputs an image ( img ), as well as source ( src ) and destination ( dst ) points. I chose the hardcode the source and destination points in the following manner:



This resulted in the following source and destination points:

I verified that my perspective transform was working as expected by drawing the src and dst points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

4. Then I use sliding window and let it iterate through `nwindow` to track the curvature. And I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:



## 3. Here is a link to my video results


## Discussion
- The first difficulty I have met is to find the best threshold for transfering the color image to binary image without losing too much information. This was solved by trying different parameters' setting.



