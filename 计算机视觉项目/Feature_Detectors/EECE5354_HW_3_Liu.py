#!/usr/bin/env python

"""
Example code for live video processing
Also multithreaded video processing sample using opencv 4.1.2

Usage:
   python Assignment_1.py {<video device number>|<video file name>}

   Use this code as a template for live video processing

   Also shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.

Keyboard shortcuts: (video display window must be selected

   ESC - exit
   space - switch between multi and single threaded processing
   a - the row derivative
   b - the column derivative
   c - the gradient magnitude
   o - the gradient angle
   d - running difference of current and previous image
   e - displays canny edges
   f - displays raw frames
   g - apply a Gaussian filter
   m - create mask for ROI
   s - do Sobel operator
   p - do Scharr operator
   l - do Laplacian operator
   h - do HSV operator
   i - do LAB operator
   v - write video output frames to file "vid_out.avi"
"""

from __future__ import print_function
from __future__ import division

import math
from collections import deque
from multiprocessing.pool import ThreadPool
from skimage.transform import pyramid_laplacian
# import the necessary packages
import cv2 as cv
import numpy as np
import scipy.ndimage as nd
import video
from common import draw_str, StatValue
from time import perf_counter, sleep
from ssc import ssc

# initialize global variables
do_gaussian = False
do_sobel = False
do_Scharr = False
do_HSV = False
do_Laplacian = False
do_LAB = False
do_labL = False
do_labA = False
do_labB = False

# Assignment 3 task
do_harris = False
do_sift = False
do_orb = False
do_gftt = False
do_nms = False

# Assignment 2 task
do_rowDerivative = False
do_colDerivative = False
do_gradientMag = False
do_gradientAngle = False

do_Krisch = False

do_LoG = False
do_Roberts = False
do_MH = False

frame_counter = 0
show_frames = True
diff_frames = False
show_edges = False

hsv_min = 0
hsv_max = 3
hsv_val = 0
mh_val = 5

lab_min = 0
lab_max = 3
lab_val = 0

canny_low_val = 0
canny_high_val = 0
canny_low = 0
canny_high = 300

vid_frames = False
ksize = 0  # variable for the size of the Sobel convolution kernel (matrix)
sqrt2ov2 = np.sqrt(2) / 2  # used to gompute Gaussian's sigma
# trackbar for Gaussian filter
int_sigma = 0  # initial value for Gaussian trackbar's sigma
gauss_slider_max = 13
# Region Of Interest variables
get_ROI = False
drag_start = (-1, -1)
now_drawing = False
rect = (0, 0), (0, 0)
have_rect = False


# used to execute process_frame when in non threaded mode
class DummyTask:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def ready():
        return True

    def get(self):
        return self.data


# this routine is run each time a new video frame is captured
def process_frame(_frame, _prevFrame, _currCount):
    global rect, have_rect, get_ROI, do_gaussian, do_Scharr, do_Laplacian, do_HSV, do_LAB, do_labA, do_rowDerivative, do_colDerivative, do_gradientMag, do_gradientAngle, canny_low_val, canny_high_val, do_LoG, do_Roberts
    # get pointers to the full size frames
    # they will only be used if have_rect is true
    # but they must be acquired before the have_rect test below 
    # because of the asynchronous mouse drag that defines the rectangle
    curr_frame = _frame
    prev_frame = _prevFrame

    # If a rectangular region is active process only in that region
    if have_rect:
        # get the coordinates of the region
        r_start, r_end = rect[0][1], rect[1][1]
        c_start, c_end = rect[0][0], rect[1][0]
        # create a pointer to the region in the full frame image 
        # reuse the names _frame and _prevFrame so that the processing 
        # steps remain the same but now they refer only to the 
        # rectangular region -- only it will be processed.
        _frame = curr_frame[slice(r_start, r_end + 1), slice(c_start, c_end + 1), :]
        _prevFrame = prev_frame[slice(r_start, r_end + 1), slice(c_start, c_end + 1), :]
        # set this flag for the end of the processing pipeline
        init_rect = True
    else:
        init_rect = False

    # If the Gaussian blur is requested, it is done first so that the Sobel and Canny operate on the blurred image
    if do_gaussian:
        _frame = GaussianFilter(_frame)  # this is a subroutine below that uses sepFilter2D
        # if not already uint8, linearly scale image into [0,255] and convert it
        if _frame.dtype != np.dtype(np.uint8):
            imin = _frame.min()
            imax = _frame.max()
            if imax > imin:
                _frame = np.uint8(255 * (_frame - imin) / (imax - imin))


    if do_harris:
        gray = cv.cvtColor(_frame, cv.COLOR_BGR2GRAY)
        corners = cv.cornerHarris(gray, 2, 3, 0.04)

        ret, dst = cv.threshold(corners, 0.04 * corners.max(), 255, cv.THRESH_TOZERO)


        if do_nms:
            kernel = np.ones((3,3), np.uint8)
            dilated = dilated = cv.dilate(dst,kernel)
            for k in range(10):
                dilated = cv.dilate(dilated,kernel)
            dst_l = np.equal(dst, dilated)
            dst = np.multiply(dst,dst_l)
        _, dst = cv.threshold(dst, 0.01 * corners.max(), 255, cv.THRESH_BINARY)
        dst = np.uint8(dst)
        ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        corners = corners[:, np.newaxis, :]
        keypoints = []
        for i in range(len(corners)):
            x, y = corners[i][0]
            strength = corners[i][0][0]
            keypoints.append(cv.KeyPoint(x=x, y=y, size=10, response=strength))

        img_keypoints = cv.drawKeypoints(_frame, keypoints, None, color=(0, 255, 0),
                                         flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        _frame = img_keypoints



    sift = cv.SIFT_create(nfeatures=100)
    if do_sift:
        gray = cv.cvtColor(_frame, cv.COLOR_BGR2GRAY)
        key, des = sift.detectAndCompute(gray, None)
        _frame = cv.drawKeypoints(_frame, key, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    orb = cv.ORB_create()
    if do_orb:
        gray = cv.cvtColor(_frame, cv.COLOR_BGR2GRAY)
        emp = np.empty((_frame.shape[0], _frame.shape[1]), dtype=np.uint8)
        key = orb.detect(gray, None)
        points, des = orb.compute(gray, key)
        if do_nms:
            cols, rows = gray.shape[::-1]
            points = ssc(key,500,0.1,cols, rows)
        _frame = cv.drawKeypoints(_frame, points, emp, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    if do_gftt:
        gray = cv.cvtColor(_frame, cv.COLOR_BGR2GRAY)
        corners = cv.goodFeaturesToTrack(gray,100,0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv.circle(_frame,(x,y),3,255,-1)


    if do_HSV:
        _frame = hsvFilter(_frame)

    if do_LAB:
        _frame = labFilter(_frame)

    if do_Scharr:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

        vert = cv.Scharr(src=blur, ddepth=cv.CV_32F, dx=1, dy=0, dst=None)
        horiz = cv.Scharr(src=blur, ddepth=cv.CV_32F, dx=0, dy=1, dst=None)
        comb = vert + horiz
        _frame = np.uint8(np.absolute(comb)).copy()

    # Assignment 2 tasks

    if do_Laplacian:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

            h8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            _frame = cv.filter2D(blur, -1, h8)
            _frame = np.uint8(np.absolute(_frame)).copy()

    if do_LoG:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

        _frame = logFilter(blur)
        # _frame = np.uint8((_frame-np.min(_frame))/(np.max(_frame)-np.min(_frame)) * 255).copy()

    if do_Roberts:
        h1 = np.array([[1, 0], [0, -1]])
        h2 = np.array([[0, 1], [-1, 0]])
        _frame1 = cv.filter2D(_frame, -1, h1)
        _frame2 = cv.filter2D(_frame, -1, h2)
        _frame = _frame1 + _frame2

    if do_sobel:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

        k1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        k2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        _frame1 = cv.filter2D(_frame, -1, k1)
        _frame2 = cv.filter2D(_frame, -1, k2)
        _frame = _frame1 + _frame2

    if do_rowDerivative:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

        _frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        row, col = np.gradient(_frame)
        _frame = np.uint8((row - np.min(row)) / (np.max(row) - np.min(row)) * 255).copy()

    if do_colDerivative:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

        _frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        row, col = np.gradient(_frame)
        _frame = np.uint8((col - np.min(col)) / (np.max(col) - np.min(col)) * 255).copy()

    if do_gradientMag:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

        _frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        row, col = np.gradient(_frame)
        _frame = np.hypot(row, col)
        _frame = np.uint8((_frame - np.min(_frame)) / (np.max(_frame) - np.min(_frame)) * 255).copy()

    if do_gradientAngle:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)

        _frame = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        row, col = np.gradient(_frame)
        _frame = np.arctan2(row, col)
        _frame = np.uint8((_frame - np.min(_frame)) / (np.max(_frame) - np.min(_frame)) * 255).copy()

    if do_Krisch:
        sigma = 1
        if do_gaussian:  # image is already blurred
            blur = _frame.copy()
        else:  # separable Gaussian is not active; use regular square neighborhood Gaussian
            # GaussianBlur is an OpenCV function; is different from Gaussian filter which is def'd below
            blur = cv.GaussianBlur(_frame, ((2 * sigma) + 1, (2 * sigma) + 1), sigma, borderType=cv.BORDER_REPLICATE)
        _frame = KirschFilter(blur)

    if do_MH:
        se3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        gray = cv.cvtColor(_frame, cv.COLOR_BGR2GRAY, _frame)
        layer = tuple(pyramid_laplacian(gray, 3, 2))
        frameList = list()
        for i in (3, 2, 1, 0):
            neg = (layer[i] <= 0).astype(np.uint8)
            pos = (layer[i] > 0).astype(np.uint8)
            dneg = cv.morphologyEx(neg, cv.MORPH_DILATE, se3, anchor=(-1, -1), borderType=cv.BORDER_REFLECT)
            v = (np.logical_and(pos, dneg)).astype(np.float64)
            v = v * layer[i]
            maxv = v.max()
            retval, v = cv.threshold(v / maxv, 0.005 * mh_val, 1, cv.THRESH_BINARY)
            frameList.append((255 * v).astype(np.uint8))

        for j in (0, 1, 2):
            shape = (int(2 * frameList[j].shape[1]), int(2 * frameList[j].shape[0]))
            newV = cv.resize(frameList[j], shape, interpolation=cv.INTER_NEAREST)
            newV = cv.morphologyEx(newV, cv.MORPH_DILATE, se3, anchor=(-1, -1), borderType=cv.BORDER_REFLECT)
            frameList[j + 1] = (255 * np.logical_and(newV, frameList[j + 1])).astype(np.uint8)

        _frame = frameList[j + 1]

    if not show_frames and show_edges:  # edges alone
        edges = cv.Canny(_frame, canny_low_val, canny_high_val)
        _frame = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
    elif show_frames and show_edges:  # edges and frames
        edges = cv.Canny(_frame, canny_low_val, canny_high_val)
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2RGB, edges)
        _frame = cv.add(_frame, edges)

    if diff_frames:
        # compute absolute difference between the current and previous frame
        difframe = cv.absdiff(_frame, _prevFrame)
        # save current frame as previous
        _prevFrame = _frame.copy()
        # set the current frame to the difference image
        _frame = difframe.copy()
    else:
        # save current frame as previous
        _prevFrame = _frame.copy()

    if have_rect and init_rect:
        # then processing has been restricted to the rectangular region
        # of the full sized images specified by rect
        # the next 4 lines may be unnecessary
        # get the coordinates of the region
        r_start, r_end = rect[0][1], rect[1][1]
        c_start, c_end = rect[0][0], rect[1][0]
        curr_frame[slice(r_start, r_end + 1), slice(c_start, c_end + 1), :] = _frame
        prev_frame[slice(r_start, r_end + 1), slice(c_start, c_end + 1), :] = _prevFrame
        # now make variables _frame and _prevframe point to 
        # the full sized images
        _frame = curr_frame
        _prevFrame = prev_frame

    return _frame, _prevFrame, _currCount


# this implementation applies a Gaussian filter using a separable kernel.
# The GaussianFilter fct needs to compute kernel size from ssigma; here's how to do that
# compute the width of the Gaussian kernel as a function of ssigma
# ksize = int(np.rint(((((ssigma - 0.8) / 0.3) + 1) / 0.5) + 1))
def GaussianFilter(_frame):
    global ksize, int_sigma, sqrt2ov2

    # if int_sigma == 0 the Gaussian is an identity, so do nothing
    if int_sigma != 0:
        ssigma = int_sigma * sqrt2ov2

        # compute the width of the Gaussian kernel as a function of sigma
        ksize = int(np.rint(((((ssigma - 0.8) / 0.3) + 1) / 0.5) + 1))
        # k has to be an odd number
        if np.mod(ksize, 2) == 0:
            ksize += 1

        # compute the Gaussian kernel with ksize support
        gkern = cv.getGaussianKernel(ksize, ssigma)
        _frame = cv.sepFilter2D(_frame, cv.CV_32F, gkern, gkern, borderType=cv.BORDER_REFLECT101)

    return _frame


def logFilter(_frame):
    global int_sigma
    _frame = cv.cvtColor(_frame, cv.COLOR_BGR2GRAY)
    _frame = nd.gaussian_laplace(_frame, 3)
    _frame = np.uint8(np.absolute(_frame)).copy()
    return _frame


def on_gauss_trackbar(val):
    global int_sigma
    int_sigma = val


def create_gaussian_trackbar():
    global int_sigma, gauss_slider_max
    cv.createTrackbar("sqrt2/2 * ", 'video', int_sigma, gauss_slider_max, on_gauss_trackbar)


def hsvFilter(_frame):
    global hsv_val

    _frame = cv.cvtColor(_frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(_frame)
    # zeros = np.zeros(h.shape, np.uint8)
    # h_img = cv.merge((h, zeros, zeros))
    # s_img = cv.merge((zeros, s, zeros))
    # v_img = cv.merge((zeros, zeros, v))
    if hsv_val == 1:
        _frame = np.uint8(np.clip((np.float32(h) / 180) * 255, 0, 255)).copy()

    if hsv_val == 2:
        _frame = np.uint8(np.clip((s / 180) * 255, 0, 255)).copy()

    if hsv_val == 3:
        _frame = np.uint8(np.clip(v, 0, 255)).copy()

    _frame = cv.applyColorMap(_frame, cv.COLORMAP_HSV)
    return _frame


def labFilter(_frame):
    global lab_val

    lab_img = cv.cvtColor(_frame, cv.COLOR_BGR2LAB)
    L, A, B = cv.split(lab_img)
    dataType = L.dtype
    CL = 100 * np.ones(np.shape(L)).astype(dataType)
    Z = np.zeros(np.shape(L)).astype(dataType)

    TA = cv.cvtColor(np.stack((CL, A, Z), axis=2), cv.COLOR_Lab2RGB)
    TB = cv.cvtColor(np.stack((CL, Z, B), axis=2), cv.COLOR_Lab2RGB)
    TL = np.uint8(np.clip((np.float32(L) / 180) * 255, 0, 255)).copy()
    TL = cv.applyColorMap(TL, cv.COLORMAP_HSV)
    if lab_val == 0:
        _frame = lab_img
    if lab_val == 1:
        _frame = TL
    if lab_val == 2:
        _frame = TA
    if lab_val == 3:
        _frame = TB
    return _frame


def on_lab_trackbar(val):
    global lab_val
    lab_val = val


def create_LAB_trackbar():
    global lab_val, lab_max
    cv.createTrackbar("LAB *", "video", lab_val, lab_max, on_lab_trackbar)


def on_hsv_trackbar(val):
    global hsv_val
    hsv_val = val


def create_HSV_trackbar():
    global hsv_min, hsv_max
    cv.createTrackbar("HSV *", "video", hsv_min, hsv_max, on_hsv_trackbar)


def on_canny_lowThreshold_trackbar(val):
    global canny_low_val
    canny_low_val = val


def on_canny_highThreshold_trackbar(val):
    global canny_high_val
    canny_high_val = val


def create_canny_trackbar():
    global canny_low, canny_high
    cv.createTrackbar("Low Threshold *", "video", canny_low, canny_high, on_canny_lowThreshold_trackbar)
    cv.createTrackbar("HIgh Threshold *", "video", canny_low, canny_high, on_canny_highThreshold_trackbar)


# onMouse call callback that draws rectangle
def onMouseGetROI(event, x, y, flags, param):
    global drag_start, rect, get_ROI, have_rect

    if event == cv.EVENT_LBUTTONDOWN:
        if get_ROI:
            print("Mouse input active")
            drag_start = (x, y)
    elif event == cv.EVENT_LBUTTONUP:
        if get_ROI:
            rect = drag_start, (x, y)
            get_ROI = False
            have_rect = True
            print("Mouse input inactive")
            print("Select rectangle deactivated")
    return rect


def KirschFilter(frame):
    k5 = np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])
    k7 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    k1 = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    k3 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    k6 = np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])
    k8 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    k4 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    k2 = np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])
    kList = [k1, k2, k3, k4, k5, k6, k7, k8]
    max = cv.filter2D(frame, -1, k1)
    for k in kList:
        _frame = cv.filter2D(frame, -1, k)
        max = np.maximum(_frame, max)

    return max


# create a video capture object
# noinspection DuplicatedCode
def create_capture(source=0):
    # parse source name (defaults to 0 which is the first USB camera attached)

    source = str(source).strip()
    chunks = source.split(':')
    # handle drive letter ('c:', ...)
    if len(chunks) > 1 and len(chunks[0]) == 1 and chunks[0].isalpha():
        chunks[1] = chunks[0] + ':' + chunks[1]
        del chunks[0]

    source = chunks[0]
    try:
        source = int(source)
    except ValueError:
        pass

    # noinspection PyTypeChecker
    params = dict(s.split('=') for s in chunks[1:])

    # video capture object defined on source

    timeout = 100
    _iter = 0
    _cap = cv.VideoCapture(source)
    while (_cap is None or not _cap.isOpened()) & (_iter < timeout):
        sleep(0.1)
        _iter = _iter + 1
        _cap = cv.VideoCapture(source)

    if _iter == timeout:
        print('camera timed out')
        return None
    else:
        print(_iter)

    if 'size' in params:
        w, h = map(int, params['size'].split('x'))
        _cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        _cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    if _cap is None or not _cap.isOpened():
        print('Warning: unable to open video source: ', source)
        return None

    return _cap


# main program
if __name__ == '__main__':
    import sys

    # print in the program shell window the text at the beginning of the file
    print(__doc__)

    # if there is no argument in the program invocation default to camera 0
    # noinspection PyBroadException
    # try:
    #     fn = sys.argv[1]
    # except:
    #     fn = 0
    if len(sys.argv) < 2:
        fn = 0
    else:
        fn = sys.argv[1]

    # grab initial frame, create window
    cv.waitKey(1) & 0xFF
    cap = video.create_capture(fn)
    ret, frame = cap.read()
    frame_counter += 1
    height, width, channels = frame.shape
    prevFrame = frame.copy()
    cv.namedWindow("video")

    # Create video of Frame sequence -- define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    vid_out = cv.VideoWriter('vid_out.avi', fourcc, 20.0, (cols, rows))

    # Set up multiprocessing
    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes=threadn)
    pending = deque()

    threaded_mode = False
    onOff = False

    # initialize time variables
    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = perf_counter()

    # main program loop
    while True:
        while len(pending) > 0 and pending[0].ready():  # there are frames in the queue
            processed_frame, prevFrame, t0 = pending.popleft().get()
            latency.update(perf_counter() - t0)
            # region of interest specified, draw the rectangle
            if have_rect:
                # draw the rectangle on the frame
                cv.rectangle(processed_frame, rect[0], rect[1], (0, 0, 255), 1)
            # plot info on threading and timing on the current image
            # comment out the next 3 lines to skip the plotting
            draw_str(processed_frame, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(processed_frame, (20, 40), "latency        :  %.1f ms" % (latency.value * 1000))
            draw_str(processed_frame, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value * 1000))
            if (do_harris or do_orb) and do_nms:
                draw_str(processed_frame, (20, 80), "NMS")
            if vid_frames:
                vid_out.write(processed_frame)
            # show the current image
            cv.imshow('video', processed_frame)

        if len(pending) < threadn:  # fewer frames than threads ==> get another frame
            # get frame
            ret, frame = cap.read()
            frame_counter += 1
            t = perf_counter()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), prevFrame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, prevFrame, t))
            pending.append(task)

        # check for a keypress
        key = cv.waitKey(1) & 0xFF

        # threaded or non threaded mode
        if key == ord(' '):
            threaded_mode = not threaded_mode

        # toggle edges
        if key == ord('e'):
            show_edges = not show_edges
            if show_edges:
                create_canny_trackbar()
            if not show_edges and do_gaussian:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                create_gaussian_trackbar()
            if not show_edges and not show_frames:
                show_frames = True

        # toggle frames
        if key == ord('f'):
            show_frames = not show_frames
            if not show_frames and not show_edges:
                show_frames = True

        # toggle Gaussian Filter
        if key == ord('g'):
            do_gaussian = not do_gaussian
            if do_gaussian:
                create_gaussian_trackbar()
            else:  # not do_gaussian:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                norm_gaussian = False

        # image difference mode
        # if key == ord('d'):
        #     diff_frames = not diff_frames

        # select / deselect rectangular ROI
        if key == ord('m'):
            # get_ROI is active only when mouse drag is active
            # have_rect is set to True after the mouse drag is complete
            # so if 'm' has been hit and have_rect is not True then activate the mouse drag
            if not have_rect:
                get_ROI = True
                print("Select rectangle activated")
                cv.setMouseCallback('video', onMouseGetROI, frame)
                # onMouseROI sets have_rect = True and get_ROI = False
            else:
                # If 'm' has been hit and have_rect is True then
                # remove the rectangle from the window and stop ROI processing
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                have_rect = False
                if do_gaussian:
                    create_gaussian_trackbar()

        # apply sobel gradient
        if key == ord('s'):
            do_sobel = not do_sobel
            if not do_sobel:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                if do_gaussian:
                    create_gaussian_trackbar()

        # ESC terminates the program
        if key == ord('v'):
            vid_frames = not vid_frames
            if vid_frames:
                print("Frames are being output to video")
            else:
                print("Frames are not being output to video")

        if key == ord('p'):
            do_Scharr = not do_Scharr
            if not do_Scharr:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                if do_gaussian:
                    create_gaussian_trackbar()

        if key == ord('l'):
            do_Laplacian = not do_Laplacian
            if not do_Laplacian:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                if do_gaussian:
                    create_gaussian_trackbar()

        if key == ord('h'):
            do_HSV = not do_HSV
            if not do_HSV:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                if do_gaussian:
                    create_gaussian_trackbar()
            if do_HSV:
                create_HSV_trackbar()

        if key == ord('i'):
            do_LAB = not do_LAB
            if not do_LAB:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)
                if do_gaussian:
                    create_gaussian_trackbar()
            if do_LAB:
                create_LAB_trackbar()

        # Assignment 3

        if key == ord('a'):
            do_harris = not do_harris
            if not do_harris:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)

        if key == ord('b'):
            do_sift = not do_sift
            if not do_sift:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)

        if key == ord('c'):
            do_orb = not do_orb
            if not do_orb:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)

        if key == ord('d'):
            do_gftt = not do_gftt
            if not do_gftt:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)

        if key == ord('n'):
            do_nms = not do_nms
            if not do_nms:
                cv.destroyWindow('video')
                cv.namedWindow('video')
                cv.imshow('video', frame)

        # if key == ord('a'):
        #     do_rowDerivative = not do_rowDerivative
        #     if not do_rowDerivative:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()
        #
        # if key == ord('b'):
        #     do_colDerivative = not do_colDerivative
        #     if not do_colDerivative:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()
        #
        # if key == ord('c'):
        #     do_gradientMag = not do_gradientMag
        #     if not do_gradientMag:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()
        #
        # if key == ord('o'):
        #     do_gradientAngle = not do_gradientAngle
        #     if not do_gradientAngle:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()
        #
        # if key == ord('k'):
        #     do_Krisch = not do_Krisch
        #     if not do_Krisch:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()
        #
        # if key == ord('q'):
        #     do_LoG = not do_LoG
        #     if not do_LoG:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()
        #
        # if key == ord('r'):
        #     do_Roberts = not do_Roberts
        #     if not do_Roberts:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()
        #
        # if key == ord('x'):
        #     do_MH = not do_MH
        #     if not do_MH:
        #         cv.destroyWindow('video')
        #         cv.namedWindow('video')
        #         cv.imshow('video', frame)
        #         if do_gaussian:
        #             create_gaussian_trackbar()

        # ESC terminates the program
        if key == 27:
            # release video capture object
            cap.release()
            # release video output object
            vid_out.release()
            cv.destroyAllWindows()
            break
