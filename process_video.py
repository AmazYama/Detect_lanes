from moviepy.editor import VideoFileClip
import pickle
import numpy as np
from cv2 import cv2
from perspective_trans import perspective_transform
from threshold_combining import combine_thresh
from Line import fit_polynomial, search_around_poly, measure_curvature_real, measure_vehicle_position
import cProfile



def lane_detection(img):
    """This is the pipeline for process one frame"""
    combine_out, gradx, grady, mag_binary, dir_binary, hls_binary = combine_thresh(img,3)
    binary_warped, M, Minv = perspective_transform(combine_out)

    #1 Using sliding window
    out_img, left_fit, right_fit = fit_polynomial(binary_warped)
    #2 Not using sliding window but the areas around the lane
    result, left_fitx, right_fitx, ploty = search_around_poly(binary_warped, left_fit, right_fit)

    # Visualization: draw the lane onto the warped blank image
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    color_warp = np.zeros((720, 1280, 3), dtype='uint8')
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (result.shape[1], result.shape[0]))
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0, dtype = cv2.CV_32F)

    # Visualization: add curvature info
    left_curve, right_curve = measure_curvature_real(left_fitx, right_fitx, ploty)
    avg_curve = (left_curve + right_curve) / 2
    label_str = 'Radius of curvature: %.1f m' % avg_curve
    result = cv2.putText(result, label_str, (30, 40), 0, 1, (0, 0, 0), 2, cv2.LINE_AA)


    # Visualization: add vehicle position info
    vehicle_position = measure_vehicle_position(result, left_fit, right_fit)
    cv2.putText(result, 'Vehicle position from lane center: %.1f m' % vehicle_position, (30, 70), 0, 1, (0, 0, 0), 2,
                cv2.LINE_AA)

    return result


if __name__=='__main__':
    clip = VideoFileClip('project_video.mp4')#.subclip(40,44)  #40,44  22,24
    clip_f = clip.fl_image(lane_detection)
    pr = cProfile.Profile()
    pr.enable()
    clip_f.write_videofile('example_images/video_test.mp4', audio = False)
    pr.disable()
    pr.print_stats(sort='time')