# Overwatch-AImbot

![altext](https://github.com/Will-J-Gale/Overwatch-AImbot/blob/master/Images/OverwatchAI_3.gif)

Overwatch AImbot detecting enemies and firing at them

## Disclaimer
This aimbot is purely experimental and intended for use in the training area only, not in real games.  
It WILL NOT work in actual games as it has only been trained on training bot images.  
It runs at less than 20fps so actual use is next to impossible anyway.  
Moreover, the limitations with vJoy make this impractical to use.  
This was created as an experiment using the Yolo V2 object detection algorithm.  

## Prerequisites 
   1. Overwatch 
   2. Xbox360ce
   3. vJoy Drivers
   2. Python 3.6
   3. OpenCV
   4. Numpy

## Yolo v2
This project utilized the Yolo V2 algorithm written in keras by:  
https://github.com/experiencor/keras-yolo2

## Downloading weights
   1. Download weigths from:  
      a. __Yolo_full_backend.h5__: https://drive.google.com/open?id=1_7dAAaY5H96JDNp_JIlc7yxmsvwDpWCx  
      b. __YOLO_Overwatch.h5__: https://drive.google.com/open?id=1jEHKwBVBss1zNgggO7G6q-UH__CnLJcF
   2. Place both of these files in the root folder (same folder as OverwatchAI.py)
   
## How to use
   1. Run Overwatch
   2. Change to windowed mode in options
   2. Change resolution to 1280x720
   3. Go into practice range
   4. Run 'OverwatchAI.py'
