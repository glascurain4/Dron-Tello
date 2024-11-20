"""
Miguel Ángel Ceniceros Fragoso A01704541
"""

import cv2
import numpy as np
from djitellopy import Tello
import math
import time

# ----------------------------------- Camera Setup

resolution = (300, 200)

#cap = cv2.VideoCapture("test.mp4")
#cap = cv2.VideoCapture(0)

tello = Tello()

tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

# ----------------------------------- BGR Color Reference

blue = (172, 26, 26)
green = (57, 158, 69)
red = (5, 0, 230)

# ----------------------------------- HSV limits

# ipad blue photo from webcam
#lower = np.array((91, 175, 251)) 
#upper = np.array((179, 255, 255))

# ipad blue photo from tello
#lower = np.array((0, 102, 143))
#upper = np.array((179, 255, 255)) 

# white calculator
#lower = np.array((0, 0, 198))
#upper = np.array((179, 255, 255))

# cima circuit from drone
# lower = np.array((30, 85, 0))
# upper = np.array((179, 255, 255))

lower = np.array((18, 57, 0))
upper = np.array((51, 255, 255))

# ----------------------------------- Region of interest

roi_width = 1 * resolution[0]
roi_height = 0.15 * resolution[1]
roi_offset = 40

left_offset = int((resolution[0] - roi_width) / 2)

roi_x0 = int((resolution[0] - roi_width) / 2)
roi_x1 = int(roi_x0 + roi_width)

roi_y0 = int(roi_offset)
roi_y1 = int(roi_y0 + roi_height)

# ----------------------------------- Drone parameters

tello_current_speed = 0
tello_speed_limit = 25 # 30 25

# --- Yaw

tello_yaw_speed_limit = 70
tello_yaw_exp_val = 0.007 # decrease for higher velocity in higher angles

# --- Lateral

tello_lateral_speed = 8 # 9
tello_lateral_val = 0.012

angle_limit = 40

# ----------------------------------- Movement functions

def get_yaw(angle):
    
    # also return forward speed
    
    if (abs(angle) > 87) or (abs(angle) < -87):
        return 0
    else:
        if (angle > 0):
            return -tello_yaw_speed_limit * np.exp(-tello_yaw_exp_val * angle)
        elif (angle < 0):
            return tello_yaw_speed_limit * np.exp(tello_yaw_exp_val * angle)
        else:
            return 0
            
def get_lateral(offset):
    
    if (abs(offset) < 25) and (abs(offset) > -25): # will depend on resolution 25
        return 0
    else:
        if (offset > 0):
            #return tello_lateral_m * offset
            return tello_lateral_speed * np.exp(tello_lateral_val * offset)
        elif (offset < 0):
            return -tello_lateral_speed * np.exp(-tello_lateral_val * offset)
        else:
            0

def drone_movement(angle, offset):
    
    f, r, y = 0, 0, 0
    
    y = int(get_yaw(angle))
    r = int(-get_lateral(offset))
    
    print(offset, r)
    #print(angle, y)
    
    #tello.send_rc_control(r, 20, -20, y)
    tello.send_rc_control(r, 24, 0, y) # 20 27

#  ----------------------------------- Main loop

path_center = (0 , 0) 

# ---- Drone takeoff

time.sleep(0.2)



while True:
    
    # --------------------------------------- Get frames
    
    #_, frame = cap.read()
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    frame = cv2.resize(frame, resolution)           # change resolution
    roi = frame[roi_y0 : roi_y1, roi_x0: roi_x1]    # get region of interest
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)      # change color space to HSV
    roi_mask = cv2.inRange(roi, lower, upper)       # get color mask
    roi_canny = cv2.Canny(roi_mask, 50, 200)        # get canny
    
    # --------------------------------------- Contours

    contours, _ = cv2.findContours(roi_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    top_points = []
    bottom_points = []
    
    for contour in contours:
        
        # ---- Get top and bottom intersections
        
        for point in contour:
            
            x, y = point[0]
            
            # if point in top bound
            if y == 0:
                top_points.append([x, y])
                cv2.circle(frame, (x + left_offset, y + roi_offset), 5, (0, 0, 255), -1)
                
            # if point in lower bound
            elif y == roi_height-1:
                bottom_points.append([x, y])
                cv2.circle(frame, (x + left_offset, y + roi_offset), 5, (0, 0, 255), -1)
        
    # If it finds two top points and two bottom points (bounding rectangle)
    
    if (len(top_points) == 2) and (len(bottom_points) == 2):
        
        # ---- For cases where the divisor is 0 -> switch to a number close to 0.
        
        calculation1 = top_points[0][0] - bottom_points[0][0]
        if calculation1 == 0:
            divisor1 = 0.1
        else:
            divisor1 = calculation1
            
        calculation2 = top_points[1][0] - bottom_points[1][0]
        if calculation2 == 0:
            divisor2 = 0.1
        else:
            divisor2 = calculation2
            
        # ---- Slopes
        
        left_slope = - (top_points[1][1] - bottom_points[1][1]) / divisor2    
        right_slope = - (top_points[0][1] - bottom_points[0][1]) / divisor1
        slope_mean = (left_slope + right_slope) / 2
        angle = math.degrees(math.atan(slope_mean))
        
        # ---- Drone offset (x axis)
        
        path_center = ((bottom_points[0][0] + top_points[1][0]) // 2, (bottom_points[0][1] + top_points[1][1]) // 2)
        drone_offset = (resolution[0] / 2) - (path_center[0] + left_offset)
        
        # ---- Drawing on screen (optional)
        
        cv2.line(frame, (bottom_points[1][0] + left_offset, bottom_points[1][1] + roi_offset), (top_points[1][0] + left_offset, top_points[1][1] + roi_offset), blue, 3) # perspective left line
        cv2.line(frame, (bottom_points[0][0] + left_offset, bottom_points[0][1] + roi_offset), (top_points[0][0] + left_offset, top_points[0][1] + roi_offset), blue, 3) # perspective right line
        cv2.circle(frame, (int(resolution[0] / 2) , int(roi_height / 2) + roi_offset), 5, blue, -1)     # draw drone center
        cv2.circle(frame, (path_center[0] + left_offset, path_center[1] + roi_offset), 5, red, -1)      # draw where drone is supposed to be
        
        # ---- Send data to movement function
        
        drone_movement(angle, drone_offset)
    
    elif (len(top_points) == 1) and (len(bottom_points) == 1): # if just one line

        
        # ---- For cases where the divisor is 0 -> switch to a number close to 0.
        
        calculation1 = top_points[0][0] - bottom_points[0][0]
        if calculation1 == 0:
            divisor1 = 0.1
        else:
            divisor1 = calculation1
                     
        # ---- Slopes
         
        line_slope = - (top_points[0][1] - bottom_points[0][1]) / divisor1
        angle = math.degrees(math.atan(line_slope))
        
        # ---- Drone offset (x axis)
        
        path_center = ((bottom_points[0][0] + top_points[0][0]) // 2, (bottom_points[0][1] + top_points[0][1]) // 2)
        drone_offset = (resolution[0] / 2) - (path_center[0] + left_offset)
        
        # ---- Drawing on screen (optional)
        
        cv2.line(frame, (bottom_points[0][0] + left_offset, bottom_points[0][1] + roi_offset), (top_points[0][0] + left_offset, top_points[0][1] + roi_offset), blue, 3) # perspective right line
        cv2.circle(frame, (int(resolution[0] / 2) , int(roi_height / 2) + roi_offset), 5, blue, -1)     # draw drone center
        cv2.circle(frame, (path_center[0] + left_offset, path_center[1] + roi_offset), 5, red, -1)      # draw where drone is supposed to be
        
        # ---- Send data to movement function
        
        drone_movement(angle, drone_offset)
        
    else:
        
        #drone_offset = (resolution[0] / 2) - (path_center[0] + left_offset)
        #drone_movement(0, drone_offset)
        
        tello.send_rc_control(0, 28, 0, 0)
        
        0
    
    
    # --------------------------------------- Show images
    
    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("ROI Mask", roi_mask)
    cv2.imshow("Canny", roi_canny)
    
    c = cv2.waitKey(1)
    if c == 27:
        tello.land()
        tello.end()
        break
    elif c == ord('t'):
        tello.takeoff()
        tello.send_rc_control(0, 0, 30, 0)
        time.sleep(2) # 2
        tello.send_rc_control(0, 0, 0, 0)
        