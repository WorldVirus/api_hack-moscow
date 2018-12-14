


import sys
import scipy.io.wavfile

sys.path.append("./OpenVokaturi-3-0a/api")
print(sys.path)
import Vokaturi

print("Loading library...")
Vokaturi.load("./OpenVokaturi-3-0a/lib/open/macos/OpenVokaturi-3-0-mac64.dylib")



def emotion_recognition(speech):


	


	
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
	    print("Neutral: %.3f" % emotionProbabilities.neutrality)
	    print("Happy: %.3f" % emotionProbabilities.happiness)
	    print("Sad: %.3f" % emotionProbabilities.sadness)
	    print("Angry: %.3f" % emotionProbabilities.anger)
	    print("Fear: %.3f" % emotionProbabilities.fear)
	else:
	    print ("Can't determine emotions")

	emotions={'Neutral':emotionProbabilities.neutrality,'Happy':emotionProbabilities.happiness, 'Sad':emotionProbabilities.sadness, 'Angry':emotionProbabilities.anger, 'Fear':emotionProbabilities.fear}
	#print(emotions)

	import operator
	emotion = max(emotions.items(), key=operator.itemgetter(1))[0]
	value = max(emotions.items(), key=operator.itemgetter(1))[1]

	print ("\n\nEmotion:\n %s with probability %.3f" % (emotion,value))


	
	
	#


    
        

    # for i in emotions:
    #     print(emotionProbabilities.i)


	voice.destroy()

	#return emotion,percent


#speech2="am.wav"

speech2="hurraurus.wav"


emotion_recognition(speech2)