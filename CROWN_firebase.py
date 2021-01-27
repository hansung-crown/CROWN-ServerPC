import os
import time
import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    print('다운로드 start')
    encoded = message.payload
    name = encoded.decode('utf-8')
    os.system('gsutil -m cp -r gs://crown-doorlock.appspot.com/' + name + ' C:\\Users\YEON\Desktop\CROWN\data')
    client.publish('crown/download/finished' , 'finished')
    print('다운로드 finished')

broker_address = '192.168.0.40'
client = mqtt.Client('firebasehi')
client.connect(broker_address, 1883)
client.subscribe('crown/download/start')
client.on_message=on_message
client.loop_forever()