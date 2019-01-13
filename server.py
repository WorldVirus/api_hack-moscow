import random,scipy
import numpy as np

from flask import Flask,session,current_app,render_template,json,jsonify,request,Response

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

#import pyaudio
import wave
from flask_socketio import emit
from flask_socketio import SocketIO, emit

RATE = 44100

cnt = 0
def write_to_file(path, data):
    sample_width = 2
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

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
app.config['FILEDIR'] = 'data_voice/'

answer =""
size_chunk = 0
cors = CORS(app)
counter = 0
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/data') # take note of this decorator syntax, it's a common pattern
@cross_origin()
def hello():
    jsonResp =  { "dialogue":["Welcome to our bank", "Hi can you help me with credit cards", "Please, waiting", "Theme: credit cards", "Routing to expert", "Its not relevant sorry bye"], "emotions":  [{"joy":"70%"},{"annoyance":"40%"}]} # test_check_data
    print(jsonify(jsonResp))
    return jsonify(jsonResp)


def get_hello():
    greeting_list = ['Ciao', 'Hei', 'Salut', 'Hola', 'Hallo', 'Hej']
    return random.choice(greeting_list)

@app.route('/postjson', methods = ['POST'])
@cross_origin()
def postJsonHandler():
    print("Request data %s",request.data)
    app.logger.debug('Body: %s', request.get_data())
    app.logger.debug("Request Headers %s", request.headers)
    print (request.is_json)
    content = request.get_json()
    print (content["speech_data"])
    f = open('./data_text/speech.txt', 'a')
    f.write(content["speech_data"] + '\n')
    f.close()
  #  main.classify_txt("finalized_model.sav","./data_text/speech.txt")
    
    #magic time!
    
    with open('./data_text/speech.txt', 'r') as f:
        req = f.readlines()[-1]
    ans = magic.predict(req) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    answer = ans

    response = app.response_class(
        response=json.dumps({'answer_value': ans}),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route('/size_taker', methods = ['POST'])
@cross_origin()
def size_request():

    print (request.is_json)
    print(request.get_json())
    audio_size = request.get_json()
    size_chunk  = audio_size["size_chunk"]
    print("SIZE: %d",size_chunk)

    return "200"
  
@app.route('/mediataker', methods = ['POST'])
@cross_origin()
def media_request():
    print("Request data %s",request.data)
    app.logger.debug('Body: %s', request.get_data())
    app.logger.debug("Request Headers %s", request.headers)
    #test = np.array(request.data)
    print("Request size_file %d",size_chunk)
    print("Request request.data %d",request.data)
    write_to_file('./data_voice/speec.wav', request.data)
    # f = open('./data_voice/speec.wav', 'w+b')
    # f.write(request.data)
    # f.close()
    path_to_wav = './data_voice/speec.wav'
    pred = speech_emotion.predict(path_to_wav)

    max_index = max(pred.items(), key=operator.itemgetter(1))[0]

    emotions[max_index] += 1
    #_count_of_messages = _count_of_messages + 1
    emotions['len'] += 1

    return "request"

@socketio.on('connect', namespace='/audio')
def test_connect():
    emit('my response', {'data': 'Herny'})

@socketio.on('start-recording', namespace='/audio')
def start_recording(options):
    """Start recording audio from the client."""
    print("hey oh")
    id = uuid.uuid4().hex  # server-side filename
    session['wavename'] = id + '.wav'
    wf = wave.open(current_app.config['FILEDIR'] + session['wavename'], 'wb')
    wf.setnchannels(options.get('numChannels', 1))
    wf.setsampwidth(options.get('bps', 16) // 8)
    wf.setframerate(options.get('fps', 44100))
    session['wavefile'] = wf

@socketio.on('write-audio', namespace='/audio')
def write_audio(data):
    """Write a chunk of audio from the client."""
    session['wavefile'].writeframes(data)

@socketio.on('end-recording', namespace='/audio')
def end_recording():
    """Stop recording audio from the client."""
    emit('add-wavefile', url_for('static',
                                 filename='_files/' + session['wavename']))
    session['wavefile'].close()
    del session['wavefile']
    del session['wavename']
    
@app.route('/emotion', methods = ['GET'])
@cross_origin()
def emotion_request():
    path_to_wav = './data_voice/her.wav'
    pred = speech_emotion.predict(path_to_wav)

    max_index = max(pred.items(), key=operator.itemgetter(1))[0]

    emotions[max_index] += 1
    #_count_of_messages = _count_of_messages + 1
    emotions['len'] += 1
    ln = emotions['len']
    for key in emotions.keys():
        emotions[key] = int(emotions[key] / ln * 100)
    print(emotions)
    response = app.response_class(
        response=json.dumps({'neutral': emotions["Neutral"]
        ,'happy':emotions["Happy"],'sad':emotions["Sad"],'angry':emotions["Angry"],'fear':emotions["Fear"],'not_enough':emotions["Not enough sonorancy to determine emotions"],'len':emotions["len"]}),
        status=200,
        mimetype='application/json'
    )
    return "request"


'''

код, который должен выполниться при окончании диалога

ln = emotions['len']
for key in emotions.keys():
    emotions[key] = int(emotions[key] / ln * 100)

emotions - словарь с процентами эмойций на диалог. Смотри выше, какие там ключи

'''

@app.route('/answer', methods = ['GET'])
def answerBot():
    jsonResp =  answer
    print(jsonify(jsonResp))
    return jsonify(jsonResp)

if __name__ == '__main__':
    app.run()