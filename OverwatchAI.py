'''
Yolo2 Keras by experiencor: https://github.com/experiencor/keras-yolo2

Overwatch Settings:
    1. Set display mode to WINDOWED
    1. Set to resolution to 1280x720
    2. Horizontal and Vertical Sensitivity: 100

x360ce Settings:
    1. Right Thumb
        a: Sensitivity: 71% (Invert checked)
'''
import os
import cv2
import numpy as np
from utils import draw_boxes
from frontend import YOLO
import json
from grabscreen import grab_screen
from vjoy import vJoy
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def sigmoid(x, scale=7):
    return (1 / (1 + math.exp(-x*scale)) - 0.5)

def moveMouse(x,y):
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_ABSOLUTE,
                         int(x/1920*65535.0), int(y/1080*65535.0))
    
def changeRange(value, minOld, maxOld, minNew, maxNew):
    oldRange = maxOld - minOld
    newRange = maxNew - minNew
    return (((value - minOld) * newRange) / oldRange) + minNew

def lookAtAndShootEnemy(controller, pos, shouldShoot):

    global UP, DOWN, LEFT, RIGHT
    
    xDir = -(CENTER[0] - pos[0]) / WIDTH
    yDir = -(CENTER[1] - pos[1]) / HEIGHT

    scale=2

    xDir = sigmoid(xDir) * scale
    yDir = sigmoid(yDir) * scale
    
##    if(xDir > 0.5):
##        xDir = 0.5
##    if(yDir > 0.5):
##        yDir = 0.5

##    if(xDir < DEAD_ZONE and xDir > 0):
##        xDir = DEAD_ZONE
##    elif(xDir < -DEAD_ZONE):
##        xDir = -DEAD_ZONE
##        
##    if(yDir < DEAD_ZONE and yDir > 0):
##        yDir = DEAD_ZONE
##    elif(yDir < -DEAD_ZONE):
##        yDir = -DEAD_ZONE
    
    if(xDir > 0.5):
        xDir = 0.5
    elif(xDir < -0.5):
        xDir = -0.5
        
    if(yDir > 0.5):
        yDir = 0.5
    elif(yDir < -0.5):
        yDir = -0.5
        
    print(xDir, yDir)

    setJoy(controller, xDir, yDir, shouldShoot, MOVE_SCALE)
  
def setJoy(controller, valueX, valueY, shouldShoot=False, scale=32768):
    #Sends move commands to virtual Xbox 360 controller
    xPos = int(changeRange(valueX, -1, 1, 0, 32768))
    yPos = int(changeRange(valueY, -1, 1, 0, 32768))
    rTrigger = 0
    
    if(shouldShoot):
        rTrigger = 32768
    
    joystickPosition = controller.generateJoystickPosition(wAxisXRot=xPos,
                                                           wAxisYRot=yPos,
                                                           wAxisZRot=rTrigger)
    controller.update(joystickPosition)

def resetController(controller):
      #Resets joystick to 0 position
      setJoy(controller, 0, 0, 0, 1)

def getClosestEnemy(enemyPositions, screenCenter):
    closestDistance = 100000000
    closestEnemy = None

    for pos in enemyPositions:
        distance = abs(screenCenter[0] - pos[0]) + abs(screenCenter[1] - pos[1])

        if(distance < closestDistance):
            closestDistance = distance
            closestEnemy = pos

    return closestEnemy, closestDistance
      
def getCenterOfBox(box, shape):
    xmin = int(box.xmin * shape[1])
    xmax = int(box.xmax * shape[1])
    ymin = int(box.ymin * shape[0])
    ymax = int(box.ymax * shape[0])

    width = abs(xmax - xmin)
    height = abs(ymax - ymin)

    centerX = xmin + (width//2)
    centerY = ymin + (height//2)

    return (centerX, centerY)

if __name__ == '__main__':
    #Yolo2 Code
    #--------------------------------------------------------------------------------------------------------------------#
    config_path  = "config.json"
    weights_path = "YOLO_Overwatch.h5"

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################


    #Custom Code
    #--------------------------------------------------------------------------------------------------------------------#
    import time
    import numpy as np

    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080
    SCREEN_CENTER_X = SCREEN_WIDTH // 2
    SCREEN_CENTER_Y = SCREEN_HEIGHT // 2
    
    WIDTH = 1031
    HEIGHT = 581
    CENTER = (WIDTH//2, HEIGHT//2)
    MOVE_SCALE = 32000
    SCREEN_REGION = (454, 240, 1484, 820)
    SHOOT_DISTANCE = 80
    
    controller = vJoy()
    controller.open()
    resetController(controller)

    DEAD_ZONE = 0.2
    index = 0

    images = []

    while(True):
        try:
            #DEBUG
            #setJoy(controller, testVal, 0, False, MOVE_SCALE)
            
            screen = grab_screen(SCREEN_REGION)
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            boxes = yolo.predict(screen)

            enemyPositions = []
            
            if(len(boxes) > 0): 
                screen = draw_boxes(screen, boxes, config['model']['labels'])
                for box in boxes:
                    centerPos = getCenterOfBox(box, screen.shape)
                    enemyPositions.append(centerPos)

                
                closestEnemy, distance = getClosestEnemy(enemyPositions, CENTER)
                cv2.circle(screen, closestEnemy, 15, (0, 0, 255), -1)
                
                shoot = False

                if(distance < SHOOT_DISTANCE):
                    shoot = True

                lookAtAndShootEnemy(controller, closestEnemy, shoot)
                
            else:
                resetController(controller)
                    
            #screen = cv2.resize(screen, (0,0), fx=0.5, fy=0.5)
            cv2.imshow("Overwatch AI", screen)
            cv2.waitKey(1)
            images.append(screen)
        except KeyboardInterrupt:
            resetController(controller)
            
            for i in range(len(images)):
                filename = f"Images/{i:05d}.png"
                cv2.imwrite(filename, images[i])
                
            break
            

