from magic.emotions import SpeechEmotionDetector
#import magic.OpenVokaturi.api.Vokaturi as Vokaturi

import operator

emotions = {
    "Neutral" : 0,
    "Happy" : 12,
    "Sad" : 34,
    "Angry" : 0,
    "Fear" : 0,
    "Not enough sonorancy to determine emotions" : 0
}

if __name__ == '__main__':

    #em = SpeechEmotionDetector('magic/OpenVokaturi/lib/open/win/OpenVokaturi-3-0-win64.dll')
    #print(em.predict('magic/OpenVokaturi/examples/hello.wav'))
    print(max(emotions.items(), key=operator.itemgetter(1))[0])