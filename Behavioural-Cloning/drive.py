# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 19:51:24 2018

@author: myidi
"""

from flask import Flask
import socketio
import eventlet
from keras.models import load_model

sio = socketio.Server()

app = Flask(__name__) #'__main__'


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    # apsp.run(port=3000)
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)