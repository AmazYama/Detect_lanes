from moviepy.editor import VideoFileClip
import pickle
import numpy as np
from cv2 import cv2
from perspective_trans import perspective_transform
from threshold_combining import combine_thresh
from Line import fit_polynomial, search_around_poly
import cProfile

def lane_detection(img):

    combine_out, gradx, grady, mag_binary, dir_binary, hls_binary = combine_thresh(img,3)
    binary_warped, M, Minv = perspective_transform(combine_out)
    #1
    out_img, left_fit, right_fit = fit_polynomial(binary_warped)
    #2
    result, left_fitx, right_fitx, ploty = search_around_poly(binary_warped, left_fit, right_fit)

    color_warp = np.zeros((720, 1280, 3), dtype='uint8')
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (result.shape[1], result.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp, 0.8, 0, dtype = cv2.CV_32F)

    return result


if __name__=='__main__':
    clip = VideoFileClip('project_video.mp4')
    clip_f = clip.fl_image(lane_detection)
    pr = cProfile.Profile()
    pr.enable()
    clip_f.write_videofile('example_images/video_test.mp4', audio = False)
    pr.disable()
    pr.print_stats(sort='time')