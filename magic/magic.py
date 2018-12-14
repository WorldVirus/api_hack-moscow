from pymorphy2 import MorphAnalyzer
import re

class MagicWorker():

    def __init__(self):
        #self.classifier = ft.load(path_to_fasttext_model)
        self.__stopwords = self.__get_stopwords__()
        self.__analyzer = MorphAnalyzer()
        self.__classes, self.__answers = self.__get_classes__()

    def __get_classes__(self):
        classes, answers = [], []
        with open('clusters.txt', 'r') as f:
            for line in f:
                tmp = list(line.split(', '))
                tmp[-1] = tmp[-1].replace('\n', '')
                classes.append(tmp[:-1])
                answers.append(tmp[-1])
        return classes, answers

    def __get_stopwords__(self):
        with open('./stopwords.txt', 'r') as f:
            stopwords = list(f.read().split('\n'))
        return stopwords

    def __process_request__(self, request : str):
        request = request.lower()
        letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя '
        spec_syms = ',./<>?;":[]{}!@#$%^&*()-=_+|'
        for sym in spec_syms:
            request = request.replace(sym, ' ')
        request = re.sub(r'\s+', ' ', request)
        request = request.replace('ё', 'е')
        result = ''
        for letter in request:
            if letter in letters:
                result += letter
        temp = []
        for word in result.split():
            temp.append(self.__analyzer.parse(word)[0].normal_form)
        result = ' '.join(temp)

        tmp_ = []
        for t in result.split(' '):
            if not (t in self.__stopwords):
                tmp_.append(t)
        result = ' '.join(tmp_) 
        return result

    def __analize__request__(self, request : str):
        processed_request = self.__process_request__(request)
        count_of_entries = [0 for _ in range(len(self.__classes))]
        for word in processed_request.split(' '):
            for i in range(len(self.__classes)):
                if word in self.__classes[i]:
                    count_of_entries[i] += 1
        persents_of_entries = [int(count_of_entries[i]/len(self.__classes[i])*100) for i in range(len(count_of_entries))]
        return persents_of_entries

    def predict(self, request : str):
        persents = self.__analize__request__(request)
        ans = 'Попробуйте переформулировать вопрос'
        max_persents_index = 0
        for i in range(1, len(persents)):
            if persents[i] > persents[max_persents_index]:
                max_persents_index = i
        if (persents[max_persents_index] > 10):# and (persents.count(persents[max_persents_index]) == 1):
            ans = self.__answers[max_persents_index]
        return ans
        
