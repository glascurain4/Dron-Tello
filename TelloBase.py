# TelloBase.py
# 11/nov/2024
# cenfra


import cv2
import time
from djitellopy import Tello



class TelloBase:
    def __init__(self, connect:bool, stream:bool, cam_source=0):

        self._drone_connect = connect
        self._drone_stream = stream

        self._drone_flying = False
        self._loop_active = True

        #self.camera_resolution = (300, 200)
        self.camera_resolution = (650, 430)

        # -------------------- setup tello -------------------- #
  
        self.tello = Tello() if self._drone_connect else None
        #self.tello = Tello()
        
        if self._drone_connect:
            self.tello.connect()

        # ------------------- set up frames ------------------- #
        
        if self._drone_connect and self._drone_stream:
            self.frame_read = self.tello.get_frame_read()
        else:
            self.frame_read = None
            self.cap = cv2.VideoCapture(cam_source)

        self.frame_function = self._get_frame_function()


    def _get_frame_drone(self):
        """Returns the frame from the drone stream."""
        return self.frame_read.frame
    

    def _get_frame_cap(self):
        """Returns the frame from the cam_source."""
        _, frame = self.cap.read()
        return frame
    

    def _get_frame_function(self):
        """Returns the reference to the function depending on the
        stream source (drone, computer webcam, video, etc)."""
        if self.frame_read:
            return self._get_frame_drone
        else:
            return self._get_frame_cap


    def _imshow(self, *args):
        """Displays frames in separate windows. Arguments should be
        tuples containing title:str and image."""
        for pair in args:
            cv2.imshow(*pair)


    def _move_drone(self, right_vel, forward_vel, up_vel, yaw_vel):
            self.tello.send_rc_control(right_vel, forward_vel, up_vel, yaw_vel)


    def _process_input(self, wait=1):
        """Process keyboard input during processing loop."""

        c = cv2.waitKey(wait)

        if c == 27: # esc: terminate everything
            if self._drone_connect:
                self.tello.land()
                self.tello.end()
            self._loop_active = False

        elif c == ord('l'): # l (L): landing
            if self._drone_connect:
                self.tello.land()

        elif c == ord('t'): # t: takeoff
            if self._drone_connect:
                self.tello.takeoff()
                self.tello.send_rc_control(0, 0, 30, 0)
                time.sleep(1) # 2
                self.tello.send_rc_control(0, 0, 0, 0)


    def _process_frame(self):        
        pass


    def MainLoop(self):
        while self._loop_active:
            self._process_frame()
