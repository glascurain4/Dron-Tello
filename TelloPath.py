# TelloPath.py
# 11/nov/2024
# cenfra


import cv2
import math
import numpy as np
from TelloBase import TelloBase


class TelloPath(TelloBase):
    def __init__(self, connect, stream, cam_source=0):

        super().__init__(connect, stream, cam_source)

        self.camera_resolution = (300, 200)

        # ----------------------- speed ----------------------- #
        
        self.forward_speed = 30
        self.lateral_speed = 8
        self.yaw_speed = 70

        self.yaw_exp = 0.007
        self.lateral_exp = 0.012

        # -------------------- hsv limits -------------------- #

        self.hsv_lower_path = np.array((91, 175, 251))
        self.hsv_upper_path = np.array((179, 255, 255))

        # ---------------- region of interest ---------------- #

        self.roi_width = 1 * self.camera_resolution[0]
        self.roi_height = int(0.15 * self.camera_resolution[1])
        
        self.roi_offset_top = 40
        self.roi_offset_left = (self.camera_resolution[0] - self.roi_width) // 2
        
        self.roi_x0 = self.roi_offset_left
        self.roi_y0 = self.roi_offset_top
        self.roi_x1 = self.roi_x0 + self.roi_width
        self.roi_y1 = self.roi_y0 + self.roi_height

        # ------------------ drawing colors ------------------ #

        self.draw_colors = {"red": (5, 0, 230),
                            "green": (57, 158, 69),
                            "blue": (172, 26, 26)}

        
    def _calculate_yaw_speed(self, angle:float):

        if (abs(angle) > 87 or abs(angle) < -87):
            return 0

        if (angle > 0):
            return -self.yaw_speed * np.exp(-self.yaw_exp * angle)
        elif (angle < 0):
            return self.yaw_speed * np.exp(self.yaw_exp * angle)
        else:
            return 0


    def _calculate_lateral_speed(self, center_offset:float):
        # this calculation will depend on the resolution of the screen.
        if (abs(center_offset) < 25 and abs(center_offset) > -25):
            return 0
        
        if (center_offset > 0):
            return self.lateral_speed * np.exp(self.lateral_exp * center_offset)
        elif (center_offset < 0):
            return -self.lateral_speed * np.exp(-self.lateral_exp * center_offset)
        else:
            return 0


    def _align_drone_with_path(self, angle:float, center_offset:float):
        """Calcualtes drone movement in order to align it with the path"""
        right_vel = round(self._calculate_lateral_speed(center_offset), 2)
        yaw_vel = round(self._calculate_yaw_speed(angle), 2)
        forward_vel = self.forward_speed
        up_vel = 0
        print("lateral:", right_vel, "yaw:", yaw_vel)
        # self._move_drone(right_vel, forward_vel, up_vel, yaw_vel)    

    
    def _get_contour_intersections(self, contours, frame):
        top_points, bottom_points = [], []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                # if point in top bound
                if y == 0:
                    top_points.append([x, y])
                    cv2.circle(frame, (x + self.roi_offset_left, y + self.roi_offset_top), 5, (0, 0, 255), -1)
                # if point in lower bound
                elif y == self.roi_height-1:
                    bottom_points.append([x, y])
                    cv2.circle(frame, (x + self.roi_offset_left, y + self.roi_offset_top), 5, (0, 0, 255), -1)
        return top_points, bottom_points
    

    def _process_points(self, top_points, bottom_points, frame):

        # fix: one line case

        num_points = min(len(top_points), len(bottom_points))
    
        if num_points == 0:
            return 0, 0

        slopes = []
        for i in range(num_points):
            calculation = top_points[i][0] - bottom_points[i][0]
            divisor = 0.1 if calculation == 0 else calculation
            slope = - (top_points[i][1] - bottom_points[i][1]) / divisor
            slopes.append(slope)

            # Drawing on screen (optional)
            cv2.line(frame, (bottom_points[i][0] + self.roi_offset_left, bottom_points[i][1] + self.roi_offset_top), 
                    (top_points[i][0] + self.roi_offset_left, top_points[i][1] + self.roi_offset_top), self.draw_colors["blue"], 3)

        if num_points == 2:
            slope_mean = sum(slopes) / num_points
            angle = math.degrees(math.atan(slope_mean))
            path_center = ((bottom_points[0][0] + top_points[1][0]) // 2, (bottom_points[0][1] + top_points[1][1]) // 2)
        else:
            angle = math.degrees(math.atan(slopes[0]))
            path_center = ((bottom_points[0][0] + top_points[0][0]) // 2, (bottom_points[0][1] + top_points[0][1]) // 2)

        center_offset = (self.camera_resolution[0] / 2) - (path_center[0] + self.roi_offset_left)

        # Drawing on screen (optional)
        cv2.circle(frame, (int(self.camera_resolution[0] / 2), int(self.roi_height / 2) + self.roi_offset_top), 5, self.draw_colors["blue"], -1)  # draw drone center
        cv2.circle(frame, (path_center[0] + self.roi_offset_left, path_center[1] + self.roi_offset_top), 5, self.draw_colors["red"], -1)  # draw where drone is supposed to be

        # Send data to movement function
        # drone_movement(angle, drone_offset)
        return angle, center_offset


    def _process_frame(self):
        """This function will be run continuously in the main loop."""
        # get frame and resize
        frame = self.frame_function()
        frame = cv2.resize(frame, self.camera_resolution)
        # process region of interest
        roi = frame[self.roi_y0 : self.roi_y1, self.roi_x0: self.roi_x1]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_mask = cv2.inRange(roi, self.hsv_lower_path, self.hsv_upper_path)
        roi_canny = cv2.Canny(roi_mask, 50, 200) 
        # get contours from region of interest
        contours, _ = cv2.findContours(roi_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # get the top and bottom points of intersection of region of interest
        top_points, bottom_points = self._get_contour_intersections(contours, frame)
        # calculate angle and offset from center
        angle, center_offset = self._process_points(top_points, bottom_points, frame)
        # use angle and offset from center to align the drone with the path
        self._align_drone_with_path(angle, center_offset)
        # show processing steps
        self._imshow(("Frame", frame), 
                        ("roi", roi), 
                        ("roi_mask", roi_mask), 
                        ("roi_canny", roi_canny))
        # process keyboard input
        self._process_input()


if __name__ == "__main__":
    instance = TelloPath(connect=False, stream=False, cam_source="tools/test.mp4")
    instance.MainLoop()