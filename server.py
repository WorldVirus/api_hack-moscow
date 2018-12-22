import random
from flask import Flask, render_template,json,jsonify,request,Response

from magic.magic import MagicWorker
from flask_cors import CORS, cross_origin

#import main
#from flask.ext.uploads import UploadSet, configure_uploads, IMAGES

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
    magic = MagicWorker()
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
    return "request"
    
@app.route('/answer', methods = ['GET'])
def answerBot():
    jsonResp =  answer
    print(jsonify(jsonResp))
    return jsonify(jsonResp)

if __name__ == '__main__':
    app.run()