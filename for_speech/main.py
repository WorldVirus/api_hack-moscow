import speech_recognition as sr #google speech to text

import pickle #for loading model

import pandas as pd
from pandas.compat import StringIO

import sys
import scipy.io.wavfile
import Vokaturi #emotion recognition


from training_classifier import lda_preprocessing ##functions for training data
from training_classifier import w2v_preprocessing ##functions for training data

def speech_to_text(speech):

	# obtain audiofile
	
	AUDIO_FILE = speech
	
	# use the audio file as the audio source
	r = sr.Recognizer()
	with sr.AudioFile(AUDIO_FILE) as source:
	    audio = r.record(source)  # read the entire audio file

	
	#  Google Cloud Speech recognition
	GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""HL2LLPTHNXTNGERGERGLLBTHKSMZDPZHJEQV"""
	try:
	    text=r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
	except sr.UnknownValueError:
	    print("Google Cloud Speech could not understand audio")
	except sr.RequestError as e:
	    print("Could not request results from Google Cloud Speech service; {0}".format(e))



	return text

def classify_txt(model, text):

	#TODO 
	# ### Predicting
	text2 = pd.read_csv(StringIO(text))


	lda_preprocessing(text2)
	w2v_preprocessing(text2)





	text2['lda_features'] = list(map(lambda doc:
	                                     document_to_lda_features(LDAmodel, doc),
	                                     text2.bow))

	text2['w2v_features'] = list(map(lambda sen_group:
	                                     get_w2v_features(W2Vmodel, sen_group),
	                                     text2.tokenized_sentences))



	X_test_lda = np.array(list(map(np.array, text2.lda_features)))
	X_test_w2v = np.array(list(map(np.array, text2.w2v_features)))
	X_test_combined = np.append(X_test_lda, X_test_w2v, axis=1)


	# ### Classification


	#prediction
	theme = model.predict_proba(X_test_combined)

	return theme


def emotion_recognition(speech):


	#import library Vokaturi
	sys.path.append("../api")
	Vokaturi.load("../lib/Vokaturi_mac.so")


	
	(sample_rate, samples) = scipy.io.wavfile.read(speech)

	buffer_length = len(samples)

	c_buffer = Vokaturi.SampleArrayC(buffer_length)
	if samples.ndim == 1:
	    c_buffer[:] = samples[:] / 32768.0  # mono
	else:
	    c_buffer[:] = 0.5*(samples[:,0]+0.0+samples[:,1]) / 32768.0  # stereo

	voice = Vokaturi.Voice (sample_rate, buffer_length)

	voice.fill(buffer_length, c_buffer)

	#extracting emotions from speech
	quality = Vokaturi.Quality()
	emotionProbabilities = Vokaturi.EmotionProbabilities()
	voice.extract(quality, emotionProbabilities)

	if quality.valid:
	    '''print("Neutral: %.3f" % emotionProbabilities.neutrality)
	    print("Happy: %.3f" % emotionProbabilities.happiness)
	    print("Sad: %.3f" % emotionProbabilities.sadness)
	    print("Angry: %.3f" % emotionProbabilities.anger)
	    print("Fear: %.3f" % emotionProbabilities.fear)'''
	    emotion = emotionProbabilities.max
	    percent = emotionProbabilities.max.percent


	voice.destroy()

	return emotion,percent


#Main 

model_name="finalized_model.sav"

# import trained model

model = pickle.load(open(model_name, 'rb'))

 
#listening for voice input

speech1=get() #Integration

txt=speech_to_text(speech1)


#get theme 
theme = classify_txt(model, txt)

#analyze emotion at the beginning

emotion1, percent1=emotion_recognition(speech1)

#Integration


#listening for last sentence

speech2=get() #Integration
emotion2, percent2=emotion_recognition(speech2)

# analyze emotions
# send to front
# integration 

send(emotion1, percent1) #integration
send(emotion2, percent2) #integration






