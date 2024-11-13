# TelloFigure.py
# 12/nov/2024
# cenfra


import cv2
import time
import numpy as np
from copy import copy
from TelloPath import TelloPath


class TelloFigure(TelloPath):
    def __init__(self, connect, stream, cam_source=0):

        super().__init__(connect, stream, cam_source)

        # ---------------------- hsv ------------------------ #

        self.hsv_lower_path = np.array([13, 85, 0])
        self.hsv_upper_path = np.array([45, 255, 255])

        self.hsv_bounds_figures = { # pairs of lower and upper
            "black": [(180, 255, 30), (180, 255, 30)],
            "white": [(0, 0, 183), (179, 18, 255)],
            "red": [(128, 127, 0), (179, 219, 255)],
            "green": [(52, 61, 218), (91, 165, 255)]
        }

        # ---------------------- figures ------------------------ #

        self.detected_figure = False
        self.detected_figure_current = None
        self.detected_figure_counter = 0

        # number of frames of consecutive detection of the same figure
        self.detected_figure_counter_limit = 10

        self.turn_right = lambda x: self.tello.rotate_clockwise(90)
        self.turn_left = lambda x: self.tello.rotate_counter_clockwise(90)
        self.turn_180 = lambda x: self.tello.rotate_clockwise(180)
        self.keep_going = lambda x: self.tello.move_forward(20)

        self.figure_actions = {
            "triangle": self.turn_right,
            "rectangle": self.turn_left,
            "circle": self.keep_going,
            "pentagon": self.turn_180,
            "star": None
        }


    def _figure_combine_color_masks(self, frame_hsv):
        """(For figure detection) Creates masks for each figure color
        and combines them into a single mask."""
        # create masks for each color
        masks = []
        for color in self.hsv_bounds_figures.keys():
            mask = cv2.inRange(frame_hsv, *self.hsv_bounds_figures[color])
            masks.append(mask)
        # combine masks into one
        combined_mask = cv2.bitwise_or(masks[0], masks[1])
        for i in range(2, len(masks)):
            combined_mask = cv2.bitwise_or(combined_mask, masks[i])
        return combined_mask
    
    
    def _figure_find_figures(self, frame_hsv, frame_drawing):
        """Finds figures in the hsv frame and returns the string of 
        the largest figure found in the frame."""

        # mask = self._figure_combine_color_masks(frame_hsv)

        #mask = self._process_figures(frame)
        mask = cv2.inRange(frame_hsv,
                                    self.hsv_bounds_figures["green"][0], 
                                    self.hsv_bounds_figures["green"][1])

        # apply filters
        mask = cv2.bilateralFilter(mask, 9, 75, 75) # works fine
        kernel = np.ones((5, 5), np.uint8) # works good
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # initialize figure variables
        figure = None
        max_figure_area = 0
        max_contour = []

        for contour in contours:

            # filter by contour area
            area = cv2.contourArea(contour)
            if area < 100: # threshold
                continue
            if area < max_figure_area:
                continue
            
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            # check if the contour is convex
            # if not cv2.isContourConvex(approx):
            #     continue

            # calculate the solidity
            # hull = cv2.convexHull(contour)
            # hull_area = cv2.contourArea(hull)
            # solidity = float(area) / hull_area
            # if solidity < 0.9: # threshold
            #     continue

            # M = cv2.moments(contour)
            # if M['m00'] != 0.0:
            #     x = int(M['m10'] / M['m00'])
            #     y = int(M['m01'] / M['m00'])
            # else:
            #     continue

            x = approx.ravel()[0]
            y = approx.ravel()[1]

            text_color = (0, 0, 255)

            vertices = len(approx)
            if vertices == 10:
                figure = "star"
            elif vertices == 3:
                figure = "triangle"
            elif vertices == 4:
                #x, y, w, h = cv2.boundingRect(approx)
                #aspect_ratio = float(w) / h
                #shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
                figure = "rectangle"
            elif vertices == 5:
                figure = "pentagon"
            elif vertices == 6:
                figure = "hexagon"
            else:
                figure = "circle"

            max_figure_area = area
            max_contour = contour
            #cv2.putText(frame_drawing, figure, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            #cv2.drawContours(frame_drawing, [approx], 0, (0, 0, 255), 5)

        if len(max_contour) != 0:
            cv2.putText(frame_drawing, figure, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.drawContours(frame_drawing, [max_contour], 0, (0, 0, 255), 5)

        self._imshow(("mask", mask))

        return figure
    

    def _figure_perform_action(self, figure:str):
        """Executes the figure's respective action."""
        if figure not in self.figure_actions.keys():
            return
        self.figure_actions[figure]()


    def _figure_process_counter(self, figure:str):
        """Handles the figure detection counters. If the 
        counter reaches a certain amount of consecutive detection
        of the same figure, performs action and resets counters."""

        # if no figure found
        if figure is None:
            self.detected_figure = False
            self.detected_figure_counter = 0
            self.detected_figure_current = None

        # if figure has just been detected
        elif figure and not self.detected_figure:
            self.detected_figure = True
            self.detected_figure_current = figure
            self.detected_figure_counter = 1
        
        # if figure was found and its the same as previous frame figure
        elif figure and self.detected_figure and (figure == self.detected_figure_current):
            self.detected_figure_counter += 1

        # if figure was found but its different than the previous detected figure
        elif figure and self.detected_figure and (figure != self.detected_figure_current):
            self.detected_figure_counter = 0
            self.detected_figure_current = figure

        # print(figure, self.detected_figure_counter)

        # if the same figure was detected in multiple frames, perform action and reset counters
        if self.detected_figure_counter >= self.detected_figure_counter_limit:
            # print("----------- detected figure:", self.detected_figure_current)
            # time.sleep(1)
            if self._drone_connect:
                self._figure_perform_action(self.detected_figure_current)
            self.detected_figure = False
            self.detected_figure_counter = 0
            self.detected_figure_current = None


    def _process_frame(self):
        """This function will be run continuously in the main loop."""
        # get frame and resize
        frame = self.frame_function()
        frame = cv2.resize(frame, self.camera_resolution)
        frame_drawing = copy(frame)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # process figures and perform actions
        figure = self._figure_find_figures(frame_hsv, frame_drawing)
        self._figure_process_counter(figure)

        # process path and align
        angle, center_offset = self._process_path(frame_hsv, frame_drawing)
        self._align_drone_with_path(angle, center_offset)

        # show processing steps
        self._imshow(("Frame", frame_drawing))
        # process keyboard input
        self._process_input()


if __name__ == "__main__":
    instance = TelloFigure(connect=False, stream=False, cam_source=0)
    instance.MainLoop()
