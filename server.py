# coding: utf-8

import random,scipy
import numpy as np

from flask import Flask,session,current_app,render_template,json,jsonify,request,Response,url_for

from magic.magic import MagicWorker
from flask_cors import CORS, cross_origin

from magic.emotions import SpeechEmotionDetector

import platform
import operator
#import main
#from flask.ext.uploads import UploadSet, configure_uploads, IMAGES
from sys import byteorder
from array import array
from struct import pack

import uuid

import wave
from flask_socketio import emit
from flask_socketio import SocketIO, emit

RATE = 44100

cnt = 0

path_to_dll = ''
if platform.system() == 'Windows':
    path_to_dll = 'magic/OpenVokaturi/lib/open/win/OpenVokaturi-3-0-win64.dll'
elif platform.system() == 'Linux':
    path_to_dll = 'magic/OpenVokaturi/lib/open/linux/OpenVokaturi-3-0-linux64.so'
else:
    path_to_dll = 'magic/OpenVokaturi/lib/open/macos/OpenVokaturi-3-0-mac64.dylib'

speech_emotion = SpeechEmotionDetector(path_to_dll)
magic = MagicWorker()

emotions = {
    "Neutral" : 0,
    "Happy" : 0,
    "Sad" : 0,
    "Angry" : 0,
    "Fear" : 0,
    "Not enough sonorancy to determine emotions" : 0,
    "len" : 0
}

_count_of_messages = 0

app = Flask(__name__)
socketio = SocketIO(app)
app.config['FILEDIR'] = 'static/data_voice/'

answer =""
size_chunk = 0
cors = CORS(app)
counter = 0
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/postjson', methods = ['POST'])
@cross_origin()
def postJsonHandler():
    print("Request data %s",request.data)
    app.logger.debug('Body: %s', request.get_data())
    app.logger.debug("Request Headers %s", request.headers)
    print (request.is_json)
    content = request.get_json()
    print (content["speech_data"])
    f = open('./static/data_text/speech.txt', 'a')
    f.write(content["speech_data"] + '\n')
    f.close()

    with open('./static/data_text/speech.txt', 'r') as f:
        req = f.readlines()[-1]
    ans = magic.predict(req) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    answer = ans

    response = app.response_class(
        response=json.dumps({'answer_value': ans}),
        status=200,
        mimetype='application/json'
    )
    return response

@socketio.on('start-recording', namespace='/audio')
def start_recording(options):
    """Start recording audio from the client."""
    print("start-recording")
    id = uuid.uuid4().hex + "" # server-side filename
    session['wavename'] = id + '.wav'
    wf = wave.open(current_app.config['FILEDIR'] + session['wavename'], 'wb')
    wf.setnchannels(options.get('numChannels', 1))
    wf.setsampwidth(options.get('bps', 16) // 8)
    wf.setframerate(options.get('fps', 44100))
    session['wavefile'] = wf

@socketio.on('write-audio', namespace='/audio')
def write_audio(data):
    print("write-audio")
    """Write a chunk of audio from the client."""
    session['wavefile'].writeframes(data)

@socketio.on('end-recording', namespace='/audio')
def end_recording():
    """Stop recording audio from the client."""
    print("end-recording")
    print(session)
    pred = speech_emotion.predict('./static/data_voice/'+session['wavename'])

    session['wavefile'].close()

    max_index = max(pred.items(), key=operator.itemgetter(1))[0]

    emotions[max_index] += 1
    #_count_of_messages = _count_of_messages + 1
    emotions['len'] += 1
    del session['wavefile']
    del session['wavename']
    
@app.route('/emotion', methods = ['GET'])
@cross_origin()
def emotion_request():
    print(emotions)
    ln = emotions['len']
    for key in emotions.keys():
        emotions[key] = int(emotions[key] / ln * 100)
    response = app.response_class(
        response=json.dumps({'neutral': emotions["Neutral"]
        ,'happy':emotions["Happy"],'sad':emotions["Sad"],'angry':emotions["Angry"],'fear':emotions["Fear"],'not_enough':emotions["Not enough sonorancy to determine emotions"],'len':emotions["len"]}),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run()