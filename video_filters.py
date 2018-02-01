from imutils.video import VideoStream
from imutils import face_utils
import datetime
import math
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import sys

predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

vs = VideoStream(usePiCamera=0).start()
time.sleep(2.0)

SCALE_FACTOR = 1
FEATHER = 11
COLOUR_CORRECT_BLUR = 0.5

#Range of facial landmarks

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
EYEBROW_TOP = 20
EYEBROW_RIGHT_TOP = 25
NOSE = 33
NOSE_RIGHT = 35
NOSE_LEFT = 31
CHEEK_LEFT = 1
MOUTH_TOP = 52
NOSE_TOP = 29
MOUTH_LEFT = 60
MOUTH_RIGHT = 54
CHEEK_RIGHT = 15
CHIN = 8
RBP = 20

POINTS = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS
ALIGN_POINTS = POINTS
OVERLAY_POINTS = [POINTS]

# loop over the frames from the video stream
# grab the frame from the threaded video stream, resize it to
# have a maximum width of 400 pixels
def get_frame():
	frame_s = vs.read()
    	frame = cv2.resize(frame_s,(800,450))
	#gray_frame = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
	#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	#frame = clahe.apply(gray_frame)
    	return frame 

#def clahe_apply(frame):

#	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
#	clahe_frame = clahe.apply(gray_frame)
#	return clahe_frame


def get_facialpoints(frame):
 
	# detect faces in the grayscale frame
	rects = detector(frame,1)
	if len(rects) == 0:
        	return -1

    	return np.matrix([[p.x, p.y] for p in predictor(frame, rects[0]).parts()])

def mark_facialpoints(frame,fpoints):
    frame = frame.copy()
    for idx, point in enumerate(fpoints):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return frame


def convex_hull(frame, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(frame, points, color=color)


def get_mask(frame, fpoints):
    frame = np.zeros(frame.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        convex_hull(frame, fpoints[group], color=1)

    frame = np.array([frame,frame,frame]).transpose((1, 2, 0))
    frame = cv2.GaussianBlur(frame, (FEATHER, FEATHER), 0) > 0
    frame = frame * 1.0
    frame = cv2.GaussianBlur(frame, (FEATHER, FEATHER), 0)

    return frame


def transform_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)

    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)

    points1 /= s1
    points2 /= s2

    u, s, vt = np.linalg.svd(points1.T * points2)
    r = (u * vt).T

    h_stack = np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T))
    return np.vstack([h_stack, np.matrix([0., 0., 1.])])


def get_im_landmarks(f):
    im = cv2.imread(f, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_facialpoints(im)

    return im, s


def warp(im, m, dshape):
    output = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im, m[:2], (dshape[1], dshape[0]), dst=output,
                   borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.WARP_INVERSE_MAP)
    return output


def color_correct(im1, im2,fpoints):
    mean_left = np.mean(fpoints[LEFT_EYE_POINTS], axis=0)
    mean_right = np.mean(fpoints[RIGHT_EYE_POINTS], axis=0)

    blur_amount = COLOUR_CORRECT_BLUR * np.linalg.norm(mean_left - mean_right)
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # avoid division errors
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


def face_swap(im1, fpoints1, im2, fpoints2):
    m = transform_points(fpoints1[ALIGN_POINTS], fpoint2[ALIGN_POINTS])

    mask = get_mask(im2, fpoints2)
    warped_mask = warp_im(mask, m, im1.shape)
    combined_mask = np.max([get_face_mask(im1, fpoints1), warped_mask], axis=0)

    warped_im2 = warp_im(im2, m, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, fpoints1)

    return im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask


def get_rotatedpoints(point, anchor, angle):
    angle_rad = math.radians(angle)
    px, py = point
    ox, oy = anchor

    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return [int(qx), int(qy)]


def blend_transparent(face, overlay):
    # BGR
    overlayim = overlay[:, :, :3]
    # A
    overlay_mask = overlay[:, :, 3:]

    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlayim * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # cast to 8 bit matrix
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def glasses_filt(glasses, should_show_bounds=True):
    face = get_frame()
    gray_frame = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
    face_clahe = clahe.apply(gray_frame)
    fpoints = get_facialpoints(face_clahe)

    p1 = np.float32([[0, 0], [599, 0], [0, 208], [599, 208]])

    if type(fpoints) is not int:
        """
        GLASSES ANCHOR POINTS:

        17 & 26 edges of left eye and right eye (left and right extrema)
        0 & 16 edges of face across eyes (other left and right extra, interpolate between 0 & 17, 16 & 26 for half way points)
        19 & 24 top of left and right brows (top extreme)
        27 is centre of the eyes on the nose (centre of glasses)
        28 is the bottom threshold of glasses (perhaps interpolate between 27 & 28 if too low) (bottom extreme)
        """

        left_extreme = [fpoints[0, 0], fpoints[0, 1]]
        right_extreme = [fpoints[16, 0], fpoints[16, 1]]
        x_diff = right_extreme[0] - left_extreme[0]
        y_diff = right_extreme[1] - left_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff, x_diff))

        # get hypotenuse
        face_width = math.sqrt((right_extreme[0] - left_extreme[0]) ** 2 +
                               (right_extreme[1] - right_extreme[1]) ** 2)
        glasses_width = face_width * 1.0

        # top and bottom of left eye
        eye_height = math.sqrt((fpoints[19, 0] - fpoints[28, 0]) ** 2 +
                               (fpoints[19, 1] - fpoints[28, 1]) ** 2)
        glasses_height = eye_height * 1.2

        glassesnew=cv2.resize(glasses,(int(glasses_width*2.25),int(glasses_height*2)),interpolation=cv2.INTER_AREA)
        # generate bounding box from the anchor points
        anchor_point = [fpoints[27, 0], fpoints[27, 1]]
        tl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tl = get_rotatedpoints(tl, anchor_point, face_angle)

        tr = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tr = get_rotatedpoints(tr, anchor_point, face_angle)

        bl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_bl = get_rotatedpoints(bl, anchor_point, face_angle)

        br = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_br = get_rotatedpoints(br, anchor_point, face_angle)

        pt = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(p1,pt)

        rot = cv2.warpPerspective(glassesnew, m, (face.shape[1], face.shape[0]))
        result_blend = blend_transparent(face, rot)

        if should_show_bounds:
            for p in pt:
                pos = (p[0], p[1])
                cv2.circle(result_blend, pos, 2, (0, 0, 255), 2)
                cv2.putText(result_blend, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))

        cv2.imshow("Glasses Filter", result_blend)
	src = result_blend
	return src;



def moustache_filt(moustache_im, should_show_bounds=False):
    face = get_frame()
    gray_frame = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
    face_clahe = clahe.apply(gray_frame)
    fpoints = get_facialpoints(face_clahe)

    p1 = np.float32([[0, 0], [599, 0], [0, 208], [599, 208]])

    """
    MOUSTACHE ANCHOR POINTS

    centre anchor point is midway between 34 (top of philtrum) and 54 (bottom of philtrum)
    width can be determined by the eyes as the mouth can move
    height also determined by the eyes as before
    generate as before and just modify multiplier coefficients & translate to anchor point?


    ^^^ mouth and jaw can move, use eyes as anchor point initially then translate to philtrum position
    """

    if type(fpoints) is not int:
        left_extreme = [fpoints[0, 0], fpoints[0, 1]]
        right_extreme = [fpoints[16, 0], fpoints[16, 1]]
        x_diff = right_extreme[0] - left_extreme[0]
        y_diff = right_extreme[1] - left_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff, x_diff))

        # get hypotenuse
        face_width = math.sqrt((right_extreme[0] - left_extreme[0]) ** 2 +
                               (right_extreme[1] - right_extreme[1]) ** 2)
        moustache_width = face_width * 0.8

        # top and bottom of left eye
        eye_height = math.sqrt((fpoints[19, 0] - fpoints[28, 0]) ** 2 +
                               (fpoints[19, 1] - fpoints[28, 1]) ** 2)
        glasses_height = eye_height * 0.8

        # generate bounding box from the anchor points
        brow_anchor = [fpoints[27, 0], fpoints[27, 1]]
        tl = [int(brow_anchor[0] - (moustache_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tl = get_rotatedpoints(tl, brow_anchor, face_angle)

        tr = [int(brow_anchor[0] + (moustache_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tr = get_rotatedpoints(tr, brow_anchor, face_angle)

        bl = [int(brow_anchor[0] - (moustache_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_bl = get_rotatedpoints(bl, brow_anchor, face_angle)

        br = [int(brow_anchor[0] + (moustache_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_br = get_rotatedpoints(br, brow_anchor, face_angle)

        # locate new location for moustache on philtrum
        top_philtrum_point = [fpoints[33, 0], fpoints[33, 1]]
        bottom_philtrum_point = [fpoints[51, 0], fpoints[51, 1]]
        philtrum_anchor = [(top_philtrum_point[0] + bottom_philtrum_point[0]) / 2,
                           (top_philtrum_point[1] + bottom_philtrum_point[1]) / 2]

        # determine distance from old origin to new origin and translate
        anchor_distance = [int(philtrum_anchor[0] - brow_anchor[0]), int(philtrum_anchor[1] - brow_anchor[1])]
        rot_tl[0] += anchor_distance[0]
        rot_tl[1] += anchor_distance[1]
        rot_tr[0] += anchor_distance[0]
        rot_tr[1] += anchor_distance[1]
        rot_bl[0] += anchor_distance[0]
        rot_bl[1] += anchor_distance[1]
        rot_br[0] += anchor_distance[0]
        rot_br[1] += anchor_distance[1]

        pt = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(p1, pt)

        rot = cv2.warpPerspective(moustache_im, m, (face.shape[1], face.shape[0]))
        result_blend = blend_transparent(face, rot)

        # annotate_landmarks(result_2, landmarks)

        if should_show_bounds:
            for p in pt:
                pos = (p[0], p[1])
                cv2.circle(result_blend, pos, 2, (0, 0, 255), 2)
                cv2.putText(result_blend, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))

        cv2.imshow("Moustache Filter", result_blend)


def face_swap_filter(swap_im, swap_im_fpoints):
    usr_im = get_frame()
    usr_im = cv2.resize(usr_im, (usr_im.shape[1] * SCALE_FACTOR, usr_im.shape[0] * SCALE_FACTOR))
    gray_frame = cv2.cvtColor(usr_im, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
    usr_imclahe = clahe.apply(gray_frame)	
    usr_fpoints = get_facialpoints(usr_imclahe)

   

    if type(usr_fpoints) is not int:
        m = transform_points(usr_fpoints[ALIGN_POINTS], swap_im_fpoints[ALIGN_POINTS])

        mask = get_mask(swap_im, swap_im_fpoints)
        warped_mask = warp(mask, m, usr_im.shape)
        combined_mask = np.max([get_mask(usr_im, usr_fpoints), warped_mask], axis=0)

        warped_swap = warp(swap_im, m, usr_im.shape)
        warped_corrected_swap = color_correct(usr_im, warped_swap, usr_fpoints)

        output = usr_im * (1.0 - combined_mask) + warped_corrected_swap * combined_mask
        cv2.imwrite("swap_output.png", output)
        out = cv2.imread("swap_output.png", 1)
        cv2.imshow("Swap Output", out)

def buzz_filter(img):
	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = w - x
		mask_height = int((h-y)*1.5)
		x1 = f_points.part(NOSE_TOP).x - ((2*mask_width)/3)
		x2 = f_points.part(NOSE_TOP).x + ((3*mask_width)/4)
		y1 = f_points.part(NOSE_TOP).y - ((2*mask_height)/3)
		y2 = f_points.part(NOSE_TOP).y + ((mask_height)/2)
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
	
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		roi = flip_im[y1:y2,x1:x2]
	
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue

	
	
	cv2.imshow("Frame",flip_im)

def panda_filter(img):
	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = w - x
		mask_height = int((h-y)*1.5)
		x1 = f_points.part(NOSE_TOP).x - ((2*mask_width)/3)
		x2 = f_points.part(NOSE_TOP).x + ((3*mask_width)/4)
		y1 = f_points.part(NOSE_TOP).y - ((2*mask_height)/3)
		y2 = f_points.part(NOSE_TOP).y + ((mask_height)/2)
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
	
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		roi = flip_im[y1:y2,x1:x2]
	
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue

	
	
	cv2.imshow("Frame",flip_im)





def bunny_ears(img):

	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = int((w - x)*3)
		mask_height = int((h-y)*1.5) 
		
		dist = int((f_points.part(CHIN).y - f_points.part(NOSE_TOP).y) * 1.5)

    		# Getting dimensions of region of interest
    		x1 = f_points.part(NOSE_TOP).x - (mask_width / 2)
    		x2 = f_points.part(NOSE_TOP).x + (mask_width / 2)
    		y1 = f_points.part(NOSE_TOP).y - (mask_height / 2) - (dist*5)/10
    		y2 = f_points.part(NOSE_TOP).y + (mask_height / 2) - (dist*5)/10
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
	
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		roi = flip_im[y1:y2,x1:x2]
	
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue

	
	
	cv2.imshow("Frame",flip_im)

def scar(img):
	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = int((w - x)*0.25)
		mask_height = int((h-y)*0.25)
		#dist = int((f_points.part(CHIN).y - f_points.part(NOSE_TOP).y) * 1.5)

    		# Getting dimensions of region of interest
    		x1 = f_points.part(RBP).x - (mask_width / 2)
    		x2 = f_points.part(RBP).x + (mask_width / 2)
    		y1 = f_points.part(RBP).y - (mask_height)
    		y2 = f_points.part(RBP).y + (0)
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
		#print roi_width
		#print roi_height
		#print mask_entire.shape
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		#print "resize img"
		#print resize_img.shape
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		#print "resize_mask"
		#print resize_mask.shape
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		#print "resize_mask_inverse"
		#print resize_mask_inverse.shape
		roi = flip_im[y1:y2,x1:x2]
		#print "roi"
		#print roi.shape
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue
	src = flip_im
	return src

def glasses_harry(face,glasses,should_show_bounds=True):
    gray_frame = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
    face_clahe = clahe.apply(gray_frame)
    fpoints = get_facialpoints(face_clahe)

    p1 = np.float32([[0, 0], [599, 0], [0, 208], [599, 208]])

    if type(fpoints) is not int:
        """
        GLASSES ANCHOR POINTS:

        17 & 26 edges of left eye and right eye (left and right extrema)
        0 & 16 edges of face across eyes (other left and right extra, interpolate between 0 & 17, 16 & 26 for half way points)
        19 & 24 top of left and right brows (top extreme)
        27 is centre of the eyes on the nose (centre of glasses)
        28 is the bottom threshold of glasses (perhaps interpolate between 27 & 28 if too low) (bottom extreme)
        """

        left_extreme = [fpoints[0, 0], fpoints[0, 1]]
        right_extreme = [fpoints[16, 0], fpoints[16, 1]]
        x_diff = right_extreme[0] - left_extreme[0]
        y_diff = right_extreme[1] - left_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff, x_diff))

        # get hypotenuse
        face_width = math.sqrt((right_extreme[0] - left_extreme[0]) ** 2 +
                               (right_extreme[1] - right_extreme[1]) ** 2)
        glasses_width = face_width * 1.0

        # top and bottom of left eye
        eye_height = math.sqrt((fpoints[19, 0] - fpoints[28, 0]) ** 2 +
                               (fpoints[19, 1] - fpoints[28, 1]) ** 2)
        glasses_height = eye_height * 1.2

        glassesnew=cv2.resize(glasses,(int(glasses_width*2.25),int(glasses_height*2)),interpolation=cv2.INTER_AREA)
        # generate bounding box from the anchor points
        anchor_point = [fpoints[27, 0], fpoints[27, 1]]
        tl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tl = get_rotatedpoints(tl, anchor_point, face_angle)

        tr = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tr = get_rotatedpoints(tr, anchor_point, face_angle)

        bl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_bl = get_rotatedpoints(bl, anchor_point, face_angle)

        br = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_br = get_rotatedpoints(br, anchor_point, face_angle)

        pt = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(p1,pt)

        rot = cv2.warpPerspective(glassesnew, m, (face.shape[1], face.shape[0]))
        result_blend = blend_transparent(face, rot)

        if should_show_bounds:
            for p in pt:
                pos = (p[0], p[1])
                cv2.circle(result_blend, pos, 2, (0, 0, 255), 2)
                cv2.putText(result_blend, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))

       
	src = result_blend
	cv2.imshow("Frame",src)

def skull(img):

	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = int((w - x))
		mask_height = int((h-y)*1.5) 
		
		# Getting dimensions of region of interest
    		x1 = f_points.part(NOSE_TOP).x - ((2*mask_width)/3)
		x2 = f_points.part(NOSE_TOP).x + ((3*mask_width)/4)
		y1 = f_points.part(NOSE_TOP).y - ((2*mask_height)/3)
		y2 = f_points.part(NOSE_TOP).y + ((mask_height)/2)
	
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
	
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		roi = flip_im[y1:y2,x1:x2]
	
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue

	
	
	cv2.imshow("Frame",flip_im)

def antlers(img):

	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = int((w - x)*150/100)
		mask_height = int((h-y)) 
		
		dist = int((f_points.part(CHIN).y - f_points.part(NOSE_TOP).y) * 1.5)

    		# Getting dimensions of region of interest
    		x1 = f_points.part(NOSE_TOP).x - (mask_width / 2)
    		x2 = f_points.part(NOSE_TOP).x + (mask_width / 2)
    		y1 = f_points.part(NOSE_TOP).y - (mask_height / 2) - (dist*120/100)
    		y2 = f_points.part(NOSE_TOP).y + (mask_height / 2) - (dist*120/100)
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
	
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		roi = flip_im[y1:y2,x1:x2]
	
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue

	
	
	cv2.imshow("Frame",flip_im)

def crown(img):

	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = int((w - x)*150/100)
		mask_height = int((h-y)*150/100) 
		
		dist = int((f_points.part(CHIN).y - f_points.part(NOSE_TOP).y) * 1.5)

    		# Getting dimensions of region of interest
    		x1 = f_points.part(NOSE_TOP).x - (mask_width / 2)
    		x2 = f_points.part(NOSE_TOP).x + (mask_width / 2)
    		y1 = f_points.part(NOSE_TOP).y - (mask_height / 2) - (dist*120/100)
    		y2 = f_points.part(NOSE_TOP).y + (mask_height / 2) - (dist*120/100)
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
	
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		roi = flip_im[y1:y2,x1:x2]
	
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue

	
	
	cv2.imshow("Frame",flip_im)




def christmas_hat(img):

	mask = img [:,:,3]
	mask_inverse = cv2.bitwise_not(mask)
	mask_entire = img[:,:,0:3]
	usr_im = get_frame()
	flip_im = cv2.flip(usr_im,1)
	gray_frame = cv2.cvtColor(flip_im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	frame = clahe.apply(gray_frame)
	faces = detector(frame,1)
	for k,d in enumerate(faces):
		f_points = predictor(frame,d)
        	#################################
		#             ROI               #
		#################################
		x,w,y,h = getFaceDim(f_points)
		#################################
		#	RESIZE MASK             #
		#################################
		mask_width = int((w - x)*2/3)
		mask_height = int((h-y)) 
		
		dist = int((f_points.part(CHIN).y - f_points.part(NOSE_TOP).y) * 1.5)

    		# Getting dimensions of region of interest
    		x1 = f_points.part(RBP).x 
    		x2 = f_points.part(EYEBROW_RIGHT_TOP).x + ((mask_width )*7/10 )
    		y1 = f_points.part(RBP).y - (mask_height / 2) - ((dist)/3)
    		y2 = f_points.part(EYEBROW_RIGHT_TOP).y + (mask_height / 2) - ((dist)/3)
	
		roi_width = x2 - x1
		roi_height = y2 - y1

		# Resizing user face and mask to apply filter
	
		resize_img = cv2.resize(mask_entire,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		#cv2.imshow('resize_img',resize_img)
		resize_mask = cv2.resize(mask,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		#cv2.imshow('resize_mask', resize_mask)
		resize_mask_inverse = cv2.resize(mask_inverse,(roi_width,roi_height),interpolation = cv2.INTER_AREA)
		#cv2.imshow('resize_mask_inverse', resize_mask_inverse)
		
		if 0xFF & cv2.waitKey(30) == 27:
			break

		roi = flip_im[y1:y2,x1:x2]
	
		try:
			flip_im = addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2)
	
		except:
			print 'Do not add mask, values are invalid'
			continue

	
	
	cv2.imshow("Frame",flip_im)
	src = flip_im
	return src

	
def bg_Overlay(src,overlay,pos=(0,0),scale = 1):
	
	#overlay = cv2.resize(overlay,(300,300))
	overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
	h,w,_ = overlay.shape  # Size of foreground
	rows,cols,_ = src.shape  # Size of background Image
	y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    	for i in range(h):
        	for j in range(w):
            		if x+i >= rows or y+j >= cols:
                		continue
            		alpha = float(overlay[i][j][3]/255) # read the alpha channel 
            		src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    	cv2.imshow("Frame", src)
	return src

def bg_Overlay1(src,overlay,pos=(0,0),scale = 1):
	
	#overlay = cv2.resize(overlay,(300,300))
	overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
	h,w,_ = overlay.shape  # Size of foreground
	rows,cols,_ = src.shape  # Size of background Image
	y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    	for i in range(h):
        	for j in range(w):
            		if x+i >= rows or y+j >= cols:
                		continue
            		alpha = float(overlay[i][j][3]/255) # read the alpha channel 
            		src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    	#cv2.imshow("Frame", src)
	return src
	
def blinking_hearts (overlay):
	
	for alpha in np.arange(0, 1.1, 0.1)[::-1]:
		source = get_frame()
		cv2.addWeighted(overlay, alpha, source, 1 - alpha,
			0,source)
		print alpha
		cv2.imshow("Frame",source)
		cv2.waitKey(125)
	
	
	
	
def getFaceDim(face_points):
	x = face_points.part(CHEEK_LEFT).x
	w = face_points.part(CHEEK_RIGHT).x
	y = face_points.part(EYEBROW_TOP).y
	h = face_points.part(CHIN).y
	return x,w,y,h

def addframeandmask(roi,resize_mask_inverse,resize_img,resize_mask,flip_im,x1,x2,y1,y2):
	background = cv2.bitwise_and(roi,roi, mask = resize_mask_inverse)
	masked_img = cv2.bitwise_and(resize_img,resize_img,mask=resize_mask)
	roi_merge = background + masked_img
	flip_im[y1:y2,x1:x2] = roi_merge
	return flip_im
		
def main():
    #c = create_capture(0)
    args = sys.argv
    should_show_bounds = False

    glasses = cv2.imread('glasses.png', -1)
    moustache = cv2.imread('moustache.png', -1)

    if "--faceswap" in args:
        if "--kate" in args:
            swap_img = cv2.imread('Kate-Winslet.jpg', -1)
            gray_swap = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)
	    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	    swap_im = clahe.apply(gray_swap)
        
        elif "--leo" in args:
            swap_img = cv2.imread('leonardo_dicaprio.jpg', -1)
	    gray_swap = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)
	    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	    swap_im = clahe.apply(gray_swap)

        elif "--cage" in args:
            swap_img = cv2.imread('nicholas_cage.jpg', -1)
            gray_swap = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)
	    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	    swap_im = clahe.apply(gray_swap)

        elif "--rai" in args:
            swap_img = cv2.imread('aishwarya_rai.jpg', -1)
            gray_swap = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)
	    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	    swap_im = clahe.apply(gray_swap)
         
        elif "--trump" in args:
            swap_img = cv2.imread('trump.jpg', -1)
            gray_swap = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)
	    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	    swap_im = clahe.apply(gray_swap)

        elif "--jLaw" in args:
            swap_img = cv2.imread('JLaw.jpg', -1)
            gray_swap = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)
	    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	    swap_im = clahe.apply(gray_swap)


		
        else:
            print("No face argument provided, options are:")
            print("--kate for Kate Winslet")
            print("--cage for Nicholas Cage")
            print("--leo for Leonardo DiCaprio")
	    print("--rai for Aishwarya Rai")
            print("--trump for Donald Trump")
	    print("--jLaw for Jennifer Lawrence")
            print("Defaulting to Trump due to a lack of arguments.")
            swap_img = cv2.imread('trump.jpg', -1)
	    gray_swap = cv2.cvtColor(swap_img, cv2.COLOR_BGR2GRAY)
	    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8,8))
	    swap_im = clahe.apply(gray_swap)

        swap_im_fpoints = get_facialpoints(swap_im)

    while True:
        if "--show-bounds" in args:
            should_show_bounds = True

        if "--glasses" in args:
            glasses_filt(glasses, should_show_bounds)
        elif "--moustache" in args:
            moustache_filt(moustache, should_show_bounds)
        elif "--faceswap" in args:
            face_swap_filter(swap_img, swap_im_fpoints)
	elif "--buzzface" in args:
	    mask = cv2.imread('Buzz.png',cv2.IMREAD_UNCHANGED)
	    buzz_filter(mask)
	elif "--pandaface" in args:
	    mask = cv2.imread('panda_face.png',cv2.IMREAD_UNCHANGED)
	    panda_filter(mask)
	elif "--bunnyears" in args:
	    mask = cv2.imread('bb_ears.png',cv2.IMREAD_UNCHANGED)
	    bunny_ears(mask)
	#elif "--scar" in args:
	    #mask = cv2.imread('harryscanresize.png',cv2.IMREAD_UNCHANGED)
	    #scar(mask)
	elif "--harrypotter" in args:
	    mask = cv2.imread('scar1.png',cv2.IMREAD_UNCHANGED)
	    src = scar(mask)
	    glasses_harry(src,glasses,should_show_bounds)
	elif "--ribbon" in args:
	    src = get_frame()
	    theme = cv2.imread('ribbon.png',cv2.IMREAD_UNCHANGED)
	    bg_Overlay(src,theme,(0,0),0.4)
	elif "--gatech" in args:
	    src = get_frame()
	    theme_wreck = cv2.imread('Wreck.png',cv2.IMREAD_UNCHANGED)
	    theme_buzz  = cv2.imread('Techbuzz.png',cv2.IMREAD_UNCHANGED)
	    theme_text  = cv2.imread('Gt_text_rw.png',cv2.IMREAD_UNCHANGED)
	    bg_Overlay(src,theme_wreck,(500,0),0.5)
	    bg_Overlay(src,theme_text,(0,0),0.5)
	    bg_Overlay(src,theme_buzz,(0,300),0.5)
	elif "--blinkinghearts" in args:
	    src = get_frame()
	    heart = cv2.imread('heart.png',cv2.IMREAD_UNCHANGED)
	    heart1 = cv2.imread('heart2.png',cv2.IMREAD_UNCHANGED)
	    heart2 = cv2.imread('violetheart.png',cv2.IMREAD_UNCHANGED)
	    heart3 = cv2.imread('yellowheart.png',cv2.IMREAD_UNCHANGED)
	    heart4 = cv2.imread('greenheart.png',cv2.IMREAD_UNCHANGED)
	    heart5 = cv2.imread('blueheart.png',cv2.IMREAD_UNCHANGED)
	    heart1 = cv2.resize(heart1,(300,300))
	    heart2 = cv2.resize(heart2,(500,500))
	    heart3 = cv2.resize(heart3,(350,350))
	    heart4 = cv2.resize(heart4,(600,600))
	    heart5 = cv2.resize(heart5,(400,400))
	    overlay = bg_Overlay1(src,heart,(600,30),0.1)
	    overlay = bg_Overlay1(src,heart,(100,300),0.1)
	    overlay = bg_Overlay1(src,heart1,(200,50),0.1)
	    overlay = bg_Overlay1(src,heart1,(500,100),0.2)
	    overlay = bg_Overlay1(src,heart1,(700,350),0.1)
	    overlay = bg_Overlay1(src,heart2,(400,200),0.1)
	    overlay = bg_Overlay1(src,heart3,(50,400),0.1)
	    overlay = bg_Overlay1(src,heart4,(95,145),0.1)
	    overlay = bg_Overlay1(src,heart5,(650,275),0.1)
	    overlay = bg_Overlay1(src,heart2,(75,368),0.1)
	    overlay = bg_Overlay1(src,heart5,(300,198),0.1)
	    overlay = bg_Overlay1(src,heart3,(30,30),0.2)
	    overlay = bg_Overlay1(src,heart5,(50,95),0.1)
	    overlay = bg_Overlay1(src,heart1,(450,300),0.1)
	    overlay = bg_Overlay1(src,heart2,(600,400),0.1)
	    

	
	    #source = get_frame()
            blinking_hearts(overlay)
	    #cv2.imshow("Frame",overlay)
	    
	elif "--antler" in args:
	    mask = cv2.imread('antler.png',cv2.IMREAD_UNCHANGED) 
	    antlers(mask)
	elif "--skull" in args:
	    mask = cv2.imread('skull.png',cv2.IMREAD_UNCHANGED)
	    skull(mask)
	elif "--christmashat" in args:
	    mask = cv2.imread('christmas_hat.png',cv2.IMREAD_UNCHANGED)
	    christmas_hat(mask)
        elif "--crown" in args:
	    mask = cv2.imread('imperial-state-crown.png',cv2.IMREAD_UNCHANGED)
	    crown(mask)
	elif "--christmastheme" in args:
	    src = get_frame()
	    mask = cv2.imread('christmas_hat.png',cv2.IMREAD_UNCHANGED)
	    text = cv2.imread('ho_ho_ho.png',cv2.IMREAD_UNCHANGED)
	    sock = cv2.imread('christmas_sock.png',cv2.IMREAD_UNCHANGED)
	    sock = cv2.resize(sock,(300,300))
	    wish = cv2.imread('xmas.png', cv2.IMREAD_UNCHANGED)
	    ginger_bread  = cv2.imread('cookie_man.png',cv2.IMREAD_UNCHANGED)
	    src=christmas_hat(mask)
	    bg_Overlay(src,text,(100,0),0.4)
	    bg_Overlay(src,wish,(600,150),0.3)
	    bg_Overlay(src,ginger_bread,(75,200),0.1)
	    bg_Overlay(src,sock,(600,300),0.5)
	
	    
        else:
            print("No arguments passed in, options are:")
            print("--glasses for glasses filter")
            print("--moustache for moustache filter")
            print("--faceswap for swapping face with anyother face(human face)")
	    print("--buzzface for swapping face with buzz")
	    print("--pandaface for swapping face with panda")
	    print("--bunnyears for getting bunny ears")
	    print("--antler")
	    print("--christmastheme")
	    print("--christmashat")
	    print("--skull")
	    print("--crown")
	    print("--harrypotter")
	    print("--gatech")
    	    print("--blinkinghearts")
	    print("--ribbon")

            break

        if 0xFF & cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()	


if __name__ == '__main__':
	main()

    

