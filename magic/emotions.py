import sys
from scipy.io import wavfile
import magic.OpenVokaturi.api.Vokaturi as Vokaturi

class SpeechEmotionDetector():

    def __init__(self, path_to_dll):
        
        Vokaturi.load(path_to_dll)
    
    def predict(self, speech_file : str):
        (sample_rate, samples) = wavfile.read(speech_file)
        buffer_length = len(samples)
        c_buffer = Vokaturi.SampleArrayC(buffer_length)
        if samples.ndim == 1:  # mono
            c_buffer[:] = samples[:] / 32768.0
        else:  # stereo
            c_buffer[:] = 0.5*(samples[:,0]+0.0+samples[:,1]) / 32768.0

        voice = Vokaturi.Voice (sample_rate, buffer_length)
        voice.fill(buffer_length, c_buffer)
        quality = Vokaturi.Quality()
        emotionProbabilities = Vokaturi.EmotionProbabilities()
        voice.extract(quality, emotionProbabilities)

        ans = dict()
        ans["Neutral"] = int(emotionProbabilities.neutrality * 100)
        ans["Happy"] = int(emotionProbabilities.happiness * 100)
        ans["Sad"] = int(emotionProbabilities.sadness * 100)
        ans["Angry"] = int(emotionProbabilities.anger * 100)
        ans["Fear"] = int(emotionProbabilities.fear * 100)
        ans["Not enough sonorancy to determine emotions"] = int(not quality.valid)

        voice.destroy()
        return ans
        
if __name__ == '__main__':
    s = SpeechEmotionDetector('magic/OpenVokaturi/lib/open/win/OpenVokaturi-3-0-win64.dll')
    print(s.predict('magic/OpenVokaturi/examples/hello.wav'))