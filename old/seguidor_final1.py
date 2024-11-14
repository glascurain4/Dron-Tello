"""
Miguel Ángel Ceniceros Fragoso A01704541
"""

import cv2
import numpy as np
from djitellopy import Tello
import math
from sympy import symbols, solve
import time

tello = Tello()

# ----------------------------------- Camera Setup

resolution = (300, 200)

#cap = cv2.VideoCapture("test.mp4")
#cap = cv2.VideoCapture(0)

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
lower = np.array((92, 175, 113)) 
upper = np.array((179, 255, 255))

# ----------------------------------- Region of interest

roi_width = 1 * resolution[0]
roi_height = 0.22 * resolution[1] # 15% 20
roi_offset = 94 # 73

min_offset = 0
max_offset = 94

min_angle = 60 # lowest angle to analyze
max_angle = 87 # almost straight path

# ------------------ Equation

slope_offset = round((min_offset - max_offset) / (min_angle - max_angle), 2)
x = symbols('x')
expr = (slope_offset * min_angle) + x - min_offset
sol = solve(expr)
line_offset = round(sol[0], 2)
print(f"{slope_offset}x + ({line_offset})")

left_offset = int((resolution[0] - roi_width) / 2)

roi_x0 = int((resolution[0] - roi_width) / 2)
roi_x1 = int(roi_x0 + roi_width)

roi_y0 = int(roi_offset)
roi_y1 = int(roi_y0 + roi_height)

# ----------------------------------- Drone parameters

tello_current_speed = 0
tello_speed_limit = 30

tello_max_forward_speed = 45

# --- Yaw

tello_yaw_speed_limit = 87 # 70 80
tello_yaw_exp_val = 0.004 # decrease for higher velocity in higher angles 0.007 0.006

# --- Lateral

tello_lateral_speed = 13 # 8 13
tello_lateral_val = 0.010 # 0.012 0.011

# --- Forward velocity



# ----------------------------------- Movement functions

def get_yaw_and_speed(angle):
    
    global roi_offset, roi_x0, roi_x1, roi_y0, roi_y1   # Modify ROI starting point
    
    if (abs(angle) > 87) or (abs(angle) < -87):         # If path is almost straight
        
        roi_offset = max_offset
        
        roi_x0 = int((resolution[0] - roi_width) / 2)
        roi_x1 = int(roi_x0 + roi_width)
        roi_y0 = int(roi_offset)
        roi_y1 = int(roi_y0 + roi_height)
        
        return 0, tello_max_forward_speed
    
    else:
        
        roi_offset = int((slope_offset * abs(angle)) + line_offset)
        if roi_offset < 0:
            roi_offset = 0
        
        roi_x0 = int((resolution[0] - roi_width) / 2)
        roi_x1 = int(roi_x0 + roi_width)
        roi_y0 = int(roi_offset)
        roi_y1 = int(roi_y0 + roi_height)
        
        if (angle > 0):
            return (-tello_yaw_speed_limit * np.exp(-tello_yaw_exp_val * angle)), (tello_max_forward_speed * np.exp(0.016 * abs(angle) - 1.4))
        elif (angle < 0):
            return (tello_yaw_speed_limit * np.exp(tello_yaw_exp_val * angle)), (tello_max_forward_speed * np.exp(0.016 * abs(angle) - 1.4))
        else:
            return 0, tello_max_forward_speed
            
def get_lateral(offset):
    
    if (abs(offset) < 20) and (abs(offset) > -20): # will depend on resolution  15 -15`
        return 0
    else:
        if (offset > 0):
            return tello_lateral_speed * np.exp(tello_lateral_val * offset)
        elif (offset < 0):
            return -tello_lateral_speed * np.exp(-tello_lateral_val * offset) 
        else:
            0

def drone_movement(angle, offset):
    
    f, r, y = 0, 0, 0
    
    y, f = get_yaw_and_speed(angle)
    r = int(-get_lateral(offset))
    
    print("Forward: %0.2f | Angle: %0.2f | Yaw: %0.2f | Offset: %0.2f | Lateral: %0.2f" % (f, angle, y, offset, r))
    #print("Forward: %0.2f | Yaw: %0.2f | Lateral: %0.2f" % (f, y, r))
    
    tello.send_rc_control(int(r), int(f), 0, int(y)) # 34

#  ----------------------------------- Main loop

path_center = (0 , 0) 

# ---- Drone takeoff

tello.takeoff()

while True:
    
    # --------------------------------------- Get frames
    
    #_, frame = cap.read()
    frame = frame_read.frame
    
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
    
    else:
        
        #drone_offset = (resolution[0] / 2) - (path_center[0] + left_offset)
        #drone_movement(0, drone_offset)
        
        tello.send_rc_control(0, 20, 0, 0)
        
        pass
    
    
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