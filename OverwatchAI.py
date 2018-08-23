import numpy as np
import os, cv2, time
import tensorflow as tf
from grabscreen import grab_screen
from getkeys import key_check
from collections import defaultdict
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from vjoy import vJoy

'''Google Object Detection API Code'''
'''_________________________________________________________________________________________________________________________________________'''

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

MODEL_NAME = 'OverwatchEnemy_Graph'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('labels', 'object-detection.pbtxt')

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, sess):
  # Get handles to input and output tensors
  ops = tf.get_default_graph().get_operations()
  all_tensor_names = {output.name for op in ops for output in op.outputs}
  tensor_dict = {}
  for key in [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes', 'detection_masks'
  ]:
    tensor_name = key + ':0'
    if tensor_name in all_tensor_names:
      tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
          tensor_name)
  if 'detection_masks' in tensor_dict:
    # The following processing is only for single image
    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes, image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(
        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
    # Follow the convention by adding back the batch dimension
    tensor_dict['detection_masks'] = tf.expand_dims(
        detection_masks_reframed, 0)
  image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

'''Custom Code'''
'''_________________________________________________________________________________________________________________________________________'''

def getEnemyPosition(boxes):
  #Generates center positions from min and max box coordinates
  #format: [ymin, xmin, ymax, xmax]
  positions = []
  
  for box in boxes:
    width = box[3] - box[1]
    height = box[2] - box[0]

    centerX = (box[1] + width / 2)
    centerY = (box[0] + height / 2)
    
    positions.append((centerX, centerY))

  return positions

def getEnemyHitLocations(boxes):
  #Generates HEAD and BODY positions from enemy positions
  hitLocations = []
  
  for box in boxes:
    width = box[3] - box[1]
    height = box[2] - box[0]

    centerX = (box[1] + width / 2)
    centerY = (box[0] + height / 2)

    headPosition = centerY - (height * 0.35)
    hitLocation = {'head': (centerX, headPosition), 'body': (centerX, centerY)}
    hitLocations.append(hitLocation)

  return hitLocations

def normToScreen(pos):
  #Changes 0-1 value to 0-width and 0-height
  x = int(pos[0] * WIDTH)
  y = int(pos[1] * HEIGHT)
  return (x, y)

def normToCVScreen(pos):
  #Changes 0-1 value to 0-scaledWidth and 0-scaledHeight
  x = int(pos[0] * SCALED_SIZE[0])
  y = int(pos[1] * SCALED_SIZE[1])
  return (x, y)

def getClosestEnemy(enemyPositions):
  closestDistance = 100000000
  closestEnemy = None

  for enemy in enemyPositions:
    pos = enemy['body']
    screenPos = normToScreen(pos)
    distance = abs(CENTER[0] - screenPos[0]) + abs(CENTER[1] - screenPos[1])
    
    if(distance < closestDistance):
      closestDistance = distance
      closestEnemy = enemy

  return closestEnemy, closestDistance

def lookAtEnemy(controller, enemyPos, shoot = False):
  #Aims at enemy
  xDir = ((enemyPos[0] * 2) - 1)
  yDir = (enemyPos[1] * 2) - 1

  setJoy(controller, xDir, yDir, MOVE_SCALE, shoot)
  #print(xDir, yDir)
  
def setJoy(controller, valueX, valueY, scale, shoot = False):
  #Sends move commands to virtual Xbox 360 controller
  if(shoot):
    triggerPos = 32768
  else:
    triggerPos = 0
    
  xPos = int(valueX*scale)
  yPos = int(valueY*scale)
  joystickPosition = controller.generateJoystickPosition(wAxisXRot = 16000+xPos, wAxisYRot = 16000+yPos, wAxisZRot = triggerPos)
  controller.update(joystickPosition)
  
def shoot(trigger = True):
  
  if(trigger):
    triggerPos = 32768
  else:
    triggerPos = 0
    
  joystickPosition = controller.generateJoystickPosition()
  controller.update(joystickPosition)

def resetController(controller):
  #Resets joystick to 0 position
  joystickPosition = controller.generateJoystickPosition()
  controller.update(joystickPosition)
  
def showFrameTime():
  global previousTime
  print(time.time() - previousTime)
  previousTime = time.time()
  
#User Variables
SCREEN_REGION = (0, 0, 1919, 1079) # Region of screen we will capture
SCREEN_SIZE = (1920, 1080)
WIDTH = 1920
HEIGHT = 1080

CENTER = (WIDTH/2, HEIGHT/2)

SCALED_SIZE = (800, 450)
SCREEN_NAME = "OverwatchAI"
previousTime = time.time()
BOT_ENABLED = False
L_DOWN = False #Press L to enable bot
THRESHOLD = 0.5 #Keep only preduictions above 50%
MOVE_SCALE = 20000 # Scales vJoy movement
enemyPositions = []
SHOOT_THRESHOLD = 200

#Controller Variables    
controller = vJoy()
controller.open()
resetController(controller)

index = 0

with detection_graph.as_default():
  with tf.Session() as sess:
    while(True):

        pressedKeys = key_check()

        if len(pressedKeys) == 0:
          L_DOWN = False
        
        for key in pressedKeys:
          if(key == "L") and not L_DOWN:
            L_DOWN = True
            BOT_ENABLED = not BOT_ENABLED
            print("Bot Enabled: {}".format(BOT_ENABLED))
            
        image_np = grab_screen(SCREEN_REGION)
        image_np = cv2.resize(image_np, SCALED_SIZE)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        if(BOT_ENABLED):

          '''Google Object Detection API Code (Make predictions from image)'''
          '''__________________________________________________________________________________________________________'''
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          output_dict = run_inference_for_single_image(image_np, sess)
          # Visualization of the results of a detection.
          
          vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks'),
              use_normalized_coordinates=True,
              line_thickness=8)
          
          '''Custom Code (Find enemy positions and aim at enemy)'''
          '''__________________________________________________________________________________________________________'''
          bestScores = np.where(output_dict['detection_scores'] > THRESHOLD)
          boxes =  [output_dict['detection_boxes'][i] for i in bestScores][0]
          
          enemyPositions = getEnemyPosition(boxes)
          hitLocations = getEnemyHitLocations(boxes)
          
          if(len(enemyPositions) > 0):
            closestEnemy, distance = getClosestEnemy(hitLocations)
            
            if(distance <= SHOOT_THRESHOLD):
              shoot = True
            else:
              shoot = False
              
            lookAtEnemy(controller, closestEnemy['head'], shoot)
            
            cv2.circle(image_np, normToCVScreen(closestEnemy['body']), 20, (0,0,255), -1) #EXPENSIVE!
            cv2.circle(image_np, normToCVScreen(closestEnemy['head']), 20, (0,0,255), -1) #EXPENSIVE!
          else:
            resetController(controller)
        else:
          resetController(controller)
          
        cv2.imshow(SCREEN_NAME, image_np) # EXPENSIVE!
        showFrameTime()

        
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
        
        
        
        
resetController(controller)

