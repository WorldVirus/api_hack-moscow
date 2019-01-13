import os
from flask import Flask, render_template
from flask_socketio import SocketIO


app = Flask(__name__)

socketio = SocketIO(app)


#app.register_blueprint(url_prefix='/audio')

#app.register_blueprint(uploads_bp, url_prefix='/uploads')
