from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle
from cv2 import cv2
from perspective_trans import perspective_transform
from threshold_combining import combine_thresh
from Line import fit_polynomial, search_around_poly
import cProfile

def lane_detection(img):
    with open('camera_cali_parameters.pickle', 'rb') as f:
        para = pickle.load(f)
    mtx = para['mtx']
    dist = para['dist']
    img = cv2.undistort(img, mtx, dist, None, None)
    combine_out, gradx, grady, mag_binary, dir_binary, hls_binary = combine_thresh(img,3)
    binary_warped, M = perspective_transform(combine_out)
    binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    # 1
    out_img, left_fit, right_fit = fit_polynomial(binary_warped)
    #2
    result, left_fitx, right_fitx, ploty = search_around_poly(binary_warped, left_fit, right_fit)
    return result


if __name__=='__main__':
    clip = VideoFileClip('project_video.mp4')

    proc_video = clip.fl_image(lane_detection(15))

    pr = cProfile.Profile()
    pr.enable()

    clip_f.write_videofile('example_images/video_test.mp4', audio = False)

    pr.disable()
    pr.print_stats(sort='time')