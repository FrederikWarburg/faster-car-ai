from picar import front_wheels, back_wheels
from picar.SunFounder_PCA9685 import Servo
import picar
from time import sleep
import cv2
import numpy as np
import picar
import os

picar.setup()


camera = cv2.VideoCapture(-1)

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

## Image Hyperparams
LWR_Threshold = np.array([0.3, 0.05, 0.70])
UPP_Threshold = np.array([0.90, 0.2, 0.9])

def image_standardize(image):
    # standardize image to mean = 0, and std = 1
    mu = np.mean(image,axis=(0,1))
    std = np.std(image,axis=(0,1))
    return ((image - mu) / std)

def image_normalized(image):
    # Scale image between 0 and 1
    min_ = np.min(image, axis=(0,1))
    max_ = np.max(image, axis=(0,1))
    return (image - min_) / (max_ - min_ )

def convert_image_to_HSV(image):
    # Convert cv2 image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (21,21), 0)
    hsv = hsv[240:,:,:]
    return hsv

def find_edges(image):
    # Find the edges based of image
    mask = cv2.inRange(image, LWR_Threshold, UPP_Threshold)
    edges = cv2.Canny(mask,200,100)
    edges = cv2.GaussianBlur(edges, (5,5), 0)
    return (edges)

def find_multiple_lines(edges):
    # Find all the edges
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100, minLineLength=100, maxLineGap=100)
    return lines

def compute_line_clusters(lines):
    # Compute clusters for each line
    from sklearn.cluster import KMeans
    lines = lines.reshape(-1,2)
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(lines)

    keypoints = np.asarray([kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]])

    lower_mean = [np.mean(keypoints[0, keypoints[1] < 120]),np.mean(keypoints[1, keypoints[1] < 120])]
    upper_mean = [np.mean(keypoints[0, keypoints[1] >= 120]),np.mean(keypoints[1, keypoints[1] >= 120])]

    return lower_mean, upper_mean

def find_line(image):
    # find the line to drive from an image in HSV
    edges = find_edges(image)
    lines = find_multiple_lines(edges)
    lower_mean, upper_mean = compute_line_clusters(lines)
    return int(lower_mean[0]), int(lower_mean[1]+240)

def main():
    print("Let's roll!")

    while True:
        # init pos
        x = 0
        y = 0

        # Get current image
        _, img = camera.read()
        #filename = 'data/img_00030.png'
        #img = cv2.imread(filename)

        img_HSV = convert_image_to_HSV(img)
        img_HSV = image_standardize(img_HSV)
        img_HSV = image_normalized(img_HSV)
        # Find line posistion
        x, y = find_line(img_HSV)
        # Drive 
        # Assuming that the camera is in the middle
        delta_x = x - 320 
        delta_y = y - 480

        # Compute turn angle based on calculated angles
        turn_angle = np.arctan(delta_x/delta_y)*180/np.pi
        fw_angle = turn_angle + 90  

        # Outcommeted Christians Turns method
        #turn_angle = int(float(CAMERA_X_ANGLE) / SCREEN_WIDTH * delta_x)
    
        
        #global pan_angle
        #pan_angle += turn_angle

        sleep(0.01)
        #fw_angle = 180 - pan_angle

        if fw_angle < FW_ANGLE_MIN or fw_angle > FW_ANGLE_MAX:
            #fw_angle = ((180 - fw_angle) - 90)/2 + 90
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
