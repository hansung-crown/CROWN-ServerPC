import cv2
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
 
from enum import Enum
from keras.models import load_model
 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
 
MODEL_PATH = "gesture_model/hand_model_gray.hdf5"
 
class GestureRecognizer:
   # GESTURE COUNT 배열 초기화
   gestureCnt = {'fist':0, 'five':0, 'point':0, 'swing':0}

   # Mapping between the output of our model and a textual representation
   CLASSES = {
       0: 'fist',
       1: 'five',
       2: 'point',
       3: 'swing'
   }

   #초기화 함수
   def __init__(self, capture=0, model_path=MODEL_PATH):
       self.bg = None
       self.frame = None # current frame
       self.kernel = np.ones((3, 3), np.uint8)
       self.recognizer = load_model(model_path)
       self.hand_bbox = (116, 116, 170, 170)
 
       # Display positions (pixel coordinates)
       self.positions = {
           'hand_pose': (15, 40), # hand pose text
           #'fps': (15, 20), # fps counter
           'null_pos': (200, 200) # used as null point for mouse control
       }
 
       # Begin capturing video
       #self.video = cv2.VideoCapture(0)
       self.video = cv2.VideoCapture('http://192.168.0.40:8090/stream/video.mjpeg')
       ok, self.frame = self.video.read()
       self.bg = self.frame.copy()

       if not self.video.isOpened():
           print("Could not open video")

	#소멸함수
   def __del__(self):
       cv2.destroyAllWindows()
       self.video.release()
 
   # Helper function for applying a mask to an array
   def mask_array(self, array, imask):
       if array.shape[:2] != imask.shape:
           raise Exception("Shapes of input and imask are incompatible")
       output = np.zeros_like(array, dtype=np.uint8)
       for i, row in enumerate(imask):
           output[i, row] = array[i, row]
       return output
 
   def extract_foreground(self):
       # Find the absolute difference between the background frame and current frame
       diff = cv2.absdiff(self.bg, self.frame)
       mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
 
       # Threshold the mask
       th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
 
       # Opening, closing and dilation
       opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
       closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
       img_dilation = cv2.dilate(closing, self.kernel, iterations=2)
 
       # Get mask indexes
       imask = img_dilation > 0
 
       # Get foreground from mask
       foreground = self.mask_array(self.frame, imask)
 
       return foreground, mask
 
   def run(self):
       # Capture, process, display loop   
       while True:
           # Read a new frame
           ok, self.frame = self.video.read()
           display = self.frame.copy() # Frame we'll do all the graphical drawing to
           data_display = np.zeros_like(display, dtype=np.uint8) # Black screen to display data
           
           if not ok:
               break
 
           if self.bg is None:
               cv2.putText(display,
                           "Press 'r' to reset the foreground extraction.",
                           self.positions['hand_pose'],
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.75, (0, 127, 64), 2)
               cv2.imshow("display", display)
 
               k = cv2.waitKey(1) & 0xff
               if k == 27: break # ESC pressed
               elif k == 114 or k == 108:
                   # r pressed
                   self.bg = self.frame.copy()
                   self.hand_bbox = (116, 116, 170, 170)
           else:
               # Extract the foreground
               foreground, mask = self.extract_foreground()
               foreground_display = foreground.copy()
 
               # Get hand from mask using the bounding box
               hand_crop = mask[int(self.hand_bbox[1]):int(self.hand_bbox[1]+self.hand_bbox[3]),
                                int(self.hand_bbox[0]):int(self.hand_bbox[0]+self.hand_bbox[2])]
 
               try:
                   # Resize cropped hand and make prediction on gesture
                   hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
                   #사전에 학습된 모델을 가지고 와서 prediction에 사용 
                   prediction = self.recognizer.predict(hand_crop_resized)
                   predi = prediction[0].argmax() # Get the index of the greatest confidence
                   gesture = self.CLASSES[predi]
                  
                   for i, pred in enumerate(prediction[0]):
                       # Draw confidence bar for each gesture
                       barx = self.positions['hand_pose'][0]
                       bary = 60 + i*60
                       bar_height = 20
                       bar_length = int(400 * pred) + barx # calculate length of confidence bar
                      
                       #print(pred)
                       # Make the most confidence prediction green
                       if i == predi:
                           colour = (0, 255, 0)
                       else:
                           colour = (0, 0, 255)
                      
                       cv2.putText(data_display,
                                   "{}: {}".format(self.CLASSES[i], pred),
                                   (self.positions['hand_pose'][0], 30 + i*60),
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   0.75, (255, 255, 255), 2)
                       cv2.rectangle(data_display,
                                     (barx, bary),
                                     (bar_length,
                                     bary - bar_height),
                                     colour,
                                     -1, 1)
                  
                       if pred > 0.6:
                           cnt = self.gestureCnt[gesture]
				            #frame이 15개 쌓였을 때 
                           if cnt == 15 :
                               print('gesture pub')
                               for i in range(len(self.CLASSES)): # 0, 1, 2, 3
                                   self.gestureCnt[self.CLASSES[i]] = 0
                               return gesture
        
                           else:
                               cnt += 1
                               print(cnt)
                               self.gestureCnt[gesture] = cnt
                               print(self.gestureCnt)
                          
                           cv2.putText(display,
                                       "hand pose: {}".format(gesture),
                                       self.positions['hand_pose'],
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       0.75, (0, 0, 255), 2)
                           cv2.putText(foreground_display,
                                       "hand pose: {}".format(gesture),
                                       self.positions['hand_pose'],
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       0.75, (0, 0, 255), 2)
               except Exception as ex:
                   cv2.putText(display,
                               "hand pose: error",
                               self.positions['hand_pose'],
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.75, (0, 0, 255), 2)
                   cv2.putText(foreground_display,
                               "hand pose: error",
                               self.positions['hand_pose'],
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.75, (0, 0, 255), 2)   
              
               # Draw bounding box
               p1 = (int(self.hand_bbox[0]), int(self.hand_bbox[1]))
               p2 = (int(self.hand_bbox[0] + self.hand_bbox[2]), int(self.hand_bbox[1] + self.hand_bbox[3]))
               cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
               cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)
 
               # Calculate difference in hand position
               hand_pos = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
               mouse_change = ((p1[0] + p2[0])//2 - self.positions['null_pos'][0], self.positions['null_pos'][0] - (p1[1] + p2[1])//2)
 
               # Draw hand moved difference
               cv2.circle(display, self.positions['null_pos'], 5, (0,0,255), -1)
               cv2.circle(display, hand_pos, 5, (0,255,0), -1)
               cv2.line(display, self.positions['null_pos'], hand_pos, (255,0,0),5)
 
               # Display pause command text
            #    cv2.putText(foreground_display,
            #                "hold 'r' to recalibrate until the screen is black",
            #                (15, 400),
            #                cv2.FONT_HERSHEY_SIMPLEX,
            #                0.65, (0, 0, 255), 2)
            #    cv2.putText(foreground_display,
            #                "to recalibrate",
            #                (15, 420),
            #                cv2.FONT_HERSHEY_SIMPLEX,
            #                0.65, (0, 0, 255), 2)
            #    cv2.putText(foreground_display,
            #                "press 'p' to return to paused state",
            #                (15, 450),
            #                cv2.FONT_HERSHEY_SIMPLEX,
            #                0.65, (0, 0, 255), 2)
 
               # Display foreground_display
               cv2.imshow("foreground_display", foreground_display)
               cv2.imshow("display", display)
               # Display result
               cv2.imshow("data", data_display)

               k = cv2.waitKey(1) & 0xff
               
               if k == 27: break # ESC pressed
               elif k == 114 or k == 108: # r pressed
                   self.bg = self.frame.copy()
                   self.hand_bbox = (116, 116, 170, 170)
               elif k == 112: # p pressed
                   # Reset to paused state
                   self.bg = None
                   cv2.destroyAllWindows()
               elif k != 255: print(k)
              