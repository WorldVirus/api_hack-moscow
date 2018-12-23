import random
from flask import Flask, render_template,json,jsonify,request,Response

from magic.magic import MagicWorker
from flask_cors import CORS, cross_origin

from magic.emotions import SpeechEmotionDetector

import platform
import operator
#import main
#from flask.ext.uploads import UploadSet, configure_uploads, IMAGES


path_to_dll = ''
if platform.system() == 'Windows':
    path_to_dll = 'magic/OpenVokaturi/lib/open/win/OpenVokaturi-3-0-win64.dll'
else:
    path_to_dll = 'magic/OpenVokaturi/lib/open/linux/OpenVokaturi-3-0-linux64.so'

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
answer =""
cors = CORS(app)
counter = 0
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/data') # take note of this decorator syntax, it's a common pattern
@cross_origin()
def hello():
    jsonResp =  { "dialogue":["Welcome to our bank", "Hi can you help me with credit cards", "Please, waiting", "Theme: credit cards", "Routing to expert", "Its not relevant sorry bye"], "emotions":  [{"joy":"70%"},{"annoyance":"40%"}]} # test_check_data
    print(jsonify(jsonResp))
    return jsonify(jsonResp)


@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
@cross_origin()
def download(filename):
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename)

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

@app.route('/mediataker', methods = ['POST'])
@cross_origin()
def media_request():
    print("Request data %s",request.data)
    app.logger.debug('Body: %s', request.get_data())
    app.logger.debug("Request Headers %s", request.headers)
    print(counter)
    f = open('./data_voice/speec.wav', 'w+b')
    f.write(request.data)
    f.close()
    path_to_wav = './data_voice/speec.wav'
    pred = speech_emotion.predict(path_to_wav)

    max_index = max(pred.items(), key=operator.itemgetter(1))[0]

    emotions[max_index] += 1
    #_count_of_messages = _count_of_messages + 1
    emotions['len'] += 1

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