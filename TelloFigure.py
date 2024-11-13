# TelloFigure.py
# 12/nov/2024
# cenfra


import cv2
import math
import numpy as np
from TelloPath import TelloPath


class TelloFigure(TelloPath):

    def __init__(self, connect, stream, cam_source=0):

        super().__init__(connect, stream, cam_source)

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
    instance = TelloFigure(connect=False, stream=False, cam_source="tools/test.mp4")
    instance.MainLoop()
