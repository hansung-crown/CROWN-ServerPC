from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import time
import copy
import json
import math
import pickle
import facenet
import argparse
import numpy as np
import detect_face
import tensorflow as tf
import paho.mqtt.client as mqtt
import matplotlib.pyplot as plt
import pyrebase
import sys
import Crypto 
sys.modules['Crypto'] = Crypto

from datetime import datetime
from scipy import misc
from sklearn.svm import SVC
from os.path import join as pjoin
from sklearn.externals import joblib
from pyfcm import FCMNotification
from Human import Human
from Known import Known
from Danger import Danger
from Unknown import Unknown
from CROWN_gesture import GestureRecognizer

gesture = b'None'
doorlockFlag = False
speakerFlag = False
key = b''
filename = ''
file =''

def known_appeared(Human, frame):
    global gesture
    Human.setRecognition(True)
    file = datetime.today().strftime("%Y%m%d%H%M%S")+'_'+Human.getGroupName()+'_'+Human.getName()+'.jpg'
    client.publish('crown/attendance', file.split('.')[0])
    print('Known : ' + Human.getName() + ' 출현!!!')
    gestureRecognizer = GestureRecognizer(1)
    gesture = gestureRecognizer.run()
    print(gesture)
    Human.setCount(0)

def danger_appeared(register_ids, Human, frame):
    if Human.getRecognition() == False:
        # storage에 순간 img 업로드
        file = datetime.today().strftime("%Y%m%d%H%M%S")+'_'+Human.getGroupName()+'_'+Human.getName()+'.jpg'
        cv2.imwrite('./logs/'+file, frame)
        uploadimg(file)
        # 안드로이드에 mqtt신호주기
        client.publish('crown/warning/danger', file.split('.')[0])
        client.publish('crown/attendance', file.split('.')[0])
        # fcm으로 알림 보내기
        fcm(register_ids,'WARNING','Danger : ' + Human.getName() + ' 출현!!!' )
        print('Danger : ' + Human.getName() + ' 출현!!!')
    Human.setRecognition(True)

def unknown_appeared(register_ids, Human, frame):
    # storage에 순간 img 업로드
    file = datetime.today().strftime("%Y%m%d%H%M%S")+'_Unknown_Unknown.jpg'
    cv2.imwrite('./logs/'+file,frame)
    uploadimg(file)
    # 안드로이드에 mqtt신호주기
    client.publish('crown/warning/unknown', file.split('.')[0])
    # fcm으로 알림 보내기
    fcm(register_ids,'WARNING','Unknown : Unknown 출현!!!' )
    print('Unknown : Unknown 출현!!!')

def gesture_successed(Human):
    global gesture
    # 안드로이드에 mqtt신호주기
    file = datetime.today().strftime("%Y%m%d%H%M%S")+'_'+Human.getGroupName()+'_'+Human.getName()+'.jpg'
    client.publish('crown/gesture/success',file.split('.')[0])
    # 도어락한테 open이라고 보내기
    client.publish('crown/doorlock', 'open')
    # 스피커한테 success라고 보내기
    client.publish('crown/speaker', 'success')
    doorlockFlag = True
    gesture = b'None'
    print(Human.getGroupName() + ' : ' + Human.getName() +' success !')

def gesture_failed(Human):
    global gesture
    # 안드로이드에 mqtt신호주기
    file = datetime.today().strftime("%Y%m%d%H%M%S")+'_'+Human.getGroupName()+'_'+Human.getName()+'.jpg'
    client.publish('crown/gesture/fail',file.split('.')[0])
    # 스피커한테 fail이라고 보내기
    client.publish('crown/speaker', 'fail')
    speakerFlag = True
    gesture = b'None'
    print(Human.getGroupName() + ' : ' + Human.getName() + ' fail !')

def uploaddb(filename, groupname, name):
    # database에 로그 upload(time아래/groupname:''/name:'')
    config = {
                "apiKey":'AIzaSyBB-iLHjGkgpFD7EJylLefeTjr57huAuJg',
                "authDomain":'crown-doorlock',
                "databaseURL":'https://crown-doorlock.firebaseio.com/',
                "storageBucket":'crown-doorlock.appspot.com'}
    firebase = pyrebase.initialize_app(config)
    db = firebase.database()
    d = {'filename': filename,
        'groupname': groupname,
        'name': name}
    data = json.dumps(d)
    uploadData = db.child('log').update(data)
    print("DB UPLOAD OK")
    
def fcm(register_ids, message_title, message_body) :
    #global register_id
    push_service = FCMNotification(api_key="AAAAy3aG8Ks:APA91bHHmMnbIXmYDlK9ve0e-pgpiLLqt9Gz29Z5JkVnNh9urRdIn6Tu7DMxIQf1IZ-I-d0oysa71bltVv8EIec0ScpKXzdKmUXul6_IduYACC2QCVl6il-5oRDDsuRMGTAc5qb4OTO9")
    result = push_service.notify_multiple_devices(registration_ids= register_ids, message_title=message_title, message_body=message_body)
    print(result)

def uploadimg (filename):
    config = {
                "apiKey":'AIzaSyBB-iLHjGkgpFD7EJylLefeTjr57huAuJg',
                "authDomain":'crown-doorlock',
                "databaseURL":'https://crown-doorlock.firebaseio.com/',
                "storageBucket":'crown-doorlock.appspot.com'}
    firebase = pyrebase.initialize_app(config)
    # storage에 이미지 upload(time/groupname/name.jpg)
    storage = firebase.storage()
    storage.child('log/'+filename).put('./logs/'+filename)

def on_message(client, userdata, message):
    global doorlockFlag
    global speakerFlag

    if message.payload == b'opened':
        doorlockFlag = True
    if message.payload == b'close':
        doorlockFlag = False
    if message.payload == b'Ididit':
        print('i did it!!!')
        speakerFlag = False

broker_address = '192.168.0.40'
client = mqtt.Client('FaceRecog')
client.connect(broker_address)
client.subscribe('crown/state/doorlock')
client.subscribe('crown/state/speaker')
client.on_message=on_message
client.loop_start()

print('Creating networks and loading parameters')
class CrownFaceRecognition(object):
    
    def __init__(self, HumanList, key, register_ids):
        self.HumanList = HumanList      
        self.stop = False
        unknown = Unknown('unknown')
        self.HumanList.append(unknown) 
        for i in self.HumanList:
            print(i.getName())
            print(i.getGroupName())
        self.key = key
        self.register_ids = register_ids
        self.filename = ''
        self.file = ''

    def setKey(self, key):
        self.key = key
        ff = open('key.txt', 'w')
        ff.write(key)
        ff.close()


    def setStop(self, stop):
        self.stop = stop

    def run(self):
        global gesture
        global doorlockFlag
        global speakerFlag

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, './')

                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                margin = 44
                frame_interval = 3
                batch_size = 1000
                image_size = 182
                input_image_size = 160

                curPeople = []

                print('Loading feature extraction model')
                modeldir = './20180402-114759/20180402-114759.pb'
                facenet.load_model(modeldir)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                classifier_filename = './CROWN_model.pkl'
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % classifier_filename_exp)

                #video_capture = cv2.VideoCapture(0)
                video_capture = cv2.VideoCapture('http://192.168.0.40:8090/stream/video.mjpeg')
                c = 0

                print('Start Recognition!')
                prevTime = 0
                start = 0

                while True:
                    if self.stop == False:
                        ret, frame = video_capture.read()
                        frame = cv2.resize(frame, (0,0), fx=1, fy=1)    #resize frame (optional)

                        curTime = time.time()    # calc fps
                        timeF = frame_interval

                        if (c % timeF == 0):
                            find_results = []

                            if frame.ndim == 2:
                                frame = facenet.to_rgb(frame)
                            frame = frame[:, :, 0:3]
                            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]

                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(frame.shape)[0:2]

                                cropped = []
                                scaled = []
                                scaled_reshape = []
                                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                                for i in range(nrof_faces):
                                    emb_array = np.zeros((1, embedding_size))

                                    bb[i][0] = det[i][0]
                                    bb[i][1] = det[i][1]
                                    bb[i][2] = det[i][2]
                                    bb[i][3] = det[i][3]

                                    # inner exception
                                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                        continue

                                    if(i>len(cropped)):
                                        #print('Running')
                                        break
                                    else:
                                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                            interpolation=cv2.INTER_CUBIC)
                                        scaled[i] = facenet.prewhiten(scaled[i])
                                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                        predictions = model.predict_proba(emb_array)
                                        best_class_indices = np.argmax(predictions, axis=1)
                                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                    #plot result idx under box
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20
                                    if best_class_probabilities > 0.70:
                                        for H_i in self.HumanList:
                                            # 이름 비교
                                            if self.HumanList[best_class_indices[0]].getName() == H_i.getName():
                                                result_names = self.HumanList[best_class_indices[0]].getName()
                                                # Danger
                                                if H_i.getGroupName() == 'Danger':
                                                    if H_i.getCount() == H_i.getMax():
                                                        danger_appeared(self.register_ids, H_i, frame)
                                                    else:
                                                        cnt = H_i.getCount()
                                                        cnt += 1
                                                        H_i.setCount(cnt)
                                                        ## result_names == DangerNames -> red!!!
                                                    curPeople.append(H_i)
                                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
                                                # Known
                                                if H_i.getGroupName() == 'Known':
                                                    # 15일때
                                                    if H_i.getCount() == H_i.getMax():
                                                        known_appeared(H_i, frame)
                                                    else:
                                                        cnt = H_i.getCount()
                                                        cnt += 1
                                                        H_i.setCount(cnt)
                                                    if gesture != b'None':
                                                        if gesture == self.key:
                                                            if doorlockFlag == False: 
                                                                gesture_successed(H_i)
                                                        else:
                                                            if speakerFlag == False:
                                                                gesture_failed(H_i)
                                                    ##result_names = KnownNames -> green!!!
                                                    curPeople.append(H_i)
                                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 255, 0), thickness=1, lineType=2)
                                    #Unknown
                                    else:
                                        if self.HumanList[len(self.HumanList)-1].getCount() == self.HumanList[len(self.HumanList)-1].getMax():
                                            if self.HumanList[len(self.HumanList)-1].getRecognition() == False:
                                                unknown_appeared(self.register_ids, self.HumanList[len(self.HumanList)-1], frame)
                                            self.HumanList[len(self.HumanList)-1].setRecognition(True)
                                        else:
                                            cnt = self.HumanList[len(self.HumanList)-1].getCount()
                                            cnt += 1
                                            self.HumanList[len(self.HumanList)-1].setCount(cnt)
                                        ## Unknown -> white!!!
                                        curPeople.append(self.HumanList[len(self.HumanList)-1])
                                        cv2.putText(frame,'Unknown', (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (255, 255, 255), thickness=1, lineType=2)
                        # 현 프레임에 인식된 사람과 전 프레임에 인식된 사람 비교해서 없으면 cnt--
                        for i in self.HumanList:
                            if i.getCount() > 0:
                                for j in curPeople:
                                    if i.getName() == j.getName():
                                        i.setCheck(True)
                                    else:
                                        i.setCheck(False)
                                if not curPeople:
                                    cnt = i.getCount()
                                    cnt -= 1
                                    i.setCount(cnt)
                                    if cnt == 0:
                                        i.setRecognition(False)
                            if i.getCheck() == False:
                                cnt = i.getCount()
                                if cnt == 0:
                                    i.setRecognition(False)
                                else:
                                    cnt -= 1
                                    i.setCount(cnt)

                        curPeople = []

                        for i in self.HumanList:
                            if i.getRecognition():
                                print(i.getName())
                                print(i.getCount())

                        sec = curTime - prevTime
                        prevTime = curTime  
                        fps = 1 / (sec)
                        str = 'FPS: %2.3f' % fps
                        text_fps_x = len(frame[0]) - 150
                        text_fps_y = 20
                        cv2.putText(frame, str, (text_fps_x, text_fps_y),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
                        cv2.imshow('Face Recognition', frame)
                    
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                    elif self.stop:
                        print('stop')
                        break
                video_capture.release()
                cv2.destroyAllWindows()
