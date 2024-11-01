import pygame
from djitellopy import Tello
import time
import cv2

class Main:
    def __init__(self):
        
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        self.clock = pygame.time.Clock()
        
        self.default_speed = 60
        self.yaw_speed = 100
        
        self.tello = Tello()
        self.tello.connect()
        print("Battery:", self.tello.get_battery())
        self.tello.streamon()
        
        self.running = True
        
    def keyboard_input(self):
        
        right, forward, up, yaw = 0, 0, 0, 0
        
        key_input = pygame.key.get_pressed()
        
        if key_input[pygame.K_t]:
            self.tello.takeoff()
            return
        
        if key_input[pygame.K_l]:
            self.tello.land()
            print("Battery:", self.tello.get_battery())
            self.running = False
            return
        
        if key_input[pygame.K_w]:
            forward = self.default_speed
        elif key_input[pygame.K_s]:
            forward = -self.default_speed
            
        if key_input[pygame.K_a]:
            right = -self.default_speed
        elif key_input[pygame.K_d]:
            right = self.default_speed
            
        if key_input[pygame.K_SPACE]:
            up = self.default_speed
        elif key_input[pygame.K_LSHIFT]:
            up = -self.default_speed
            
        if key_input[pygame.K_LEFT]:
            yaw = -self.yaw_speed
        elif key_input[pygame.K_RIGHT]:
            yaw = self.yaw_speed
            
        print(right, forward, up, yaw)
            
        self.tello.send_rc_control(right, forward, up, yaw)
        
    def get_image(self):

        img = self.tello.get_frame_read().frame
        img = cv2.resize(img, (360, 240))
        cv2.imshow("Tello Cam", img)
        cv2.waitKey(1)
        
        
    def loop(self):
        while self.running:
            self.keyboard_input()
            #self.get_image()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            pygame.display.flip()
            self.clock.tick(60)
        
        
        

instance = Main()
instance.loop()
