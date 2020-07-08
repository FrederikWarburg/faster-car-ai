from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os

picar.setup()


img = cv2.VideoCapture(-1)

bw = back_wheels.Back_Wheels()
fw = front_wheels.Front_Wheels()

picar.setup()
fw.offset = 0
bw.speed = 0
fw.turn(90)


SCREEN_WIDTH = 160*5
SCREEN_HIGHT = 120*5
img.set(3,SCREEN_WIDTH)
img.set(4,SCREEN_HIGHT)
CENTER_X = SCREEN_WIDTH/2
CENTER_Y = SCREEN_HIGHT/2
CAMERA_X_ANGLE = 20
CAMERA_Y_ANGLE = 20
FW_ANGLE_MAX    = 90+30
FW_ANGLE_MIN    = 90-30
pan_angle = 90              # initial angle for pan
rear_wheels_enable  = True
front_wheels_enable = True

motor_speed = 60

def find_line():
    return CENTER_X, CENTER_Y

def main():
    print("Let's roll!")

    while True:
        # init pos
        x = 0
        y = 0

        # Find line posistion
        x, y = find_line()

        # Drive 
        delta_x = CENTER_X - x
        delta_y = CENTER_Y - y
        turn_angle = int(float(CAMERA_X_ANGLE) / SCREEN_WIDTH * delta_x)

        global pan_angle
        pan_angle += turn_angle

        sleep(0.01)
        fw_angle = 180 - pan_angle

        if fw_angle < FW_ANGLE_MIN or fw_angle > FW_ANGLE_MAX:
            fw_angle = ((180 - fw_angle) - 90)/2 + 90
            if front_wheels_enable:
                fw.turn(fw_angle)
            if rear_wheels_enable:
                bw.speed = motor_speed
                bw.backward()
            else:
                if front_wheels_enable:
                    fw.turn(fw_angle)
                if rear_wheels_enable:
                    bw.speed = motor_speed
                    bw.forward()
        else:
            bw.speed = motor_speed
            bw.forward()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        bw.stop()
        img.release()
