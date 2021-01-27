import os
import sys
import time
import paho.mqtt.client as mqtt

from Human import Human
from Known import Known
from Danger import Danger
from Unknown import Unknown
from CROWN_aligndata import Align_data
from CROWN_classifier import Classifiy_data
from CROWN_face import CrownFaceRecognition

HumanList = []
HumanNameList = []
name = ''
groupName = ''
downloadFlag = False
gestureKey = b''
register_ids = ['efLmnaKDjYI:APA91bFA_HIUONhi_dY9rJXqiShCyK9y7JXoZvAkg_2qSGzL4PEKsKbzBQ8j_Qs2lHldbj0KCLUg0bFGD7PurZWy6A2Rfb9_TP72UahHmyLxQ8CkuiGuiuOFb_cCZWCgiv11vFBPkd3N']

#시작할 때 기존에 저장되어 있는 key를 가져온다
def openKey():
    global gestureKey
    f  = open('key.txt','r')
    gestureKey = f.readline()
    f.close()

# 시작할 때 기존에 저장되어 있는 HumanList를 가져온다
def openHumanList():
    global HumanList

    f = open('HumanList.txt','r')
    lines = f.readlines()

    for line in lines:
        line = line[:-1]
        name = line.split('/')[0]
        groupName = line.split('/')[1]
        if groupName=='Known':
            HumanList.append(Known(name))
        if groupName=='Danger':
            HumanList.append(Danger(name))
        HumanNameList.append(name)
    f.close()

#  새로운 인물을 HumanList.txt에 추가한다
def appendHumanList(name, groupName):
    global HumanNameList

    data = name + '/' + groupName
    ff = open('HumanList.txt', 'a+')

    if name not in HumanNameList:
        ff.write(data)
        ff.write('\n')
    ff.close()

# MQTT
def on_message(client, userdata, message):
    global name
    global groupName
    global HumanList
    global downloadFlag
    global gestureKey
    global register_ids

    if message.topic == 'crown/addgroup':
        # realtime face 중단시키기
        faceRecog.setStop(True)
        encoded = message.payload
        name = encoded.decode('utf-8').split('/')[1]
        groupName = encoded.decode('utf-8').split('/')[0]
        size = len(HumanList)
        del HumanList[size-1]
        appendHumanList(name, groupName)
        client.publish('crown/download/start', name)
        print('다운로드 시작')
    elif message.topic == 'crown/download/finished':
        print("flag 바꾸기 전")
        downloadFlag = True
    # token 받으면
    elif message.topic == 'crown/token':
        encoded = message.payload
        register_id = encoded.decode('utf-8')
    # gesture key 받으면 key 변경
    elif message.topic == 'crown/gesture/key':
        encoded = message.payload
        key = encoded.decode('utf-8')
        print('gesture key : ' + key)
        faceRecog.setKey(key)

broker_address = '192.168.0.40'
client = mqtt.Client('crown')
client.connect(broker_address, 1883)
client.subscribe('crown/addgroup')
client.subscribe('crown/download/finished')
client.subscribe('crown/gesture/key')
client.subscribe('crown/token')
client.on_message=on_message
client.loop_start() 

if __name__ == '__main__':
    openHumanList()
    openKey()
    HumanList.sort(key = lambda object:object.name)

    faceRecog = CrownFaceRecognition(HumanList, gestureKey, register_ids)
    faceRecog.run()

    while True:
        if downloadFlag == True:
            Align_data.run('1')
            Classifiy_data.run('1')
            downloadFlag = False
            HumanList = []
            HumanNameList = []
            openHumanList()
            HumanList.sort(key = lambda object:object.name)
            faceRecog = CrownFaceRecognition(HumanList, gestureKey, register_ids)
            faceRecog.run()