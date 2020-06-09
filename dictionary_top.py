from transformers import BertTokenizer
import logging
import udpipe
import spacy
import pymorphy2
import numpy as np
from scipy.special import softmax
module_logger = logging.getLogger("graduateWork.dictionary_top")


class Dictionary:
    def __init__(self, path: str):
        """Constructor"""
        self._name = path
        file = open(path, 'r')
        dict = set()
        for line in file:
            dict.add(line.rstrip("\n"))
        self._dictionary = dict

    def __str__(self):
        stroka = "Словарь: " + self._name + "\n"
        for word in self._dictionary:
            stroka += word + "|"
        return stroka


class PredictionsForWord:
    """docstring"""
    _vocabulary = []
    _morph = None

    def __init__(self, word: str):
        """Constructor"""
        self._word = word
        self._bert = {}
        self._normalized_bert = {}
        self._skipgram = {}
        self._fasttext = {}
        self._bert_dictionary = {}
        self._skipgram_dictionary = {}
        self._fasttext_dictionary = {}
        if PredictionsForWord._morph == None:
            PredictionsForWord._morph = pymorphy2.MorphAnalyzer()



    def __str__(self):
        stroka = "------------------------------------------"
        stroka += "Словарь предсказания для слова: " + self._word + "\n"
        stroka += "Список словарей: \n"
        for dict in PredictionsForWord._vocabulary:
            stroka += str(dict) + "\n"
        stroka += "Предсказания бертом: \n" + str(self._bert) + "\n"
        stroka += "Предсказания skipgram: \n" + str(self._skipgram) + "\n"
        stroka += "Предсказания fasttext: \n" + str(self._fasttext) + "\n"
        stroka += "Включение слов из словарей в выдаче берта: \n" + str(self._bert_dictionary) + "\n"
        stroka += "Включение слов из словарей в выдаче skipgram: \n" + str(self._skipgram_dictionary) + "\n"
        stroka += "Включение слов из словарей в выдаче fasttext: \n" + str(self._fasttext_dictionary) + "\n"
        stroka += "------------------------------------------"
        return stroka

    def get_bert(self, word: str):
        logger = logging.getLogger("graduateWork.dictionary_top.get_bert")
        cell = self._bert.get(word)
        if cell == None:
            logger.info("Word: %s not found in bert" % (cell))
        return cell

    def get_normalized_bert(self, word: str):
        logger = logging.getLogger("graduateWork.dictionary_top.get_normalized_bert")
        cell = self._normalized_bert.get(word)
        if cell == None:
            logger.info("Word: %s not found in normalized bert" % (cell))
        return cell

    def get_skipgram(self, word: str):
        logger = logging.getLogger("graduateWork.dictionary_top.get_skipgram")
        cell = self._skipgram.get(word)
        if cell == None:
            logger.info("Word: %s not found in skipgram" % (cell))
        return cell

    def get_fasttext(self, word: str):
        logger = logging.getLogger("graduateWork.dictionary_top.get_fasttext")
        cell = self._fasttext.get(word)
        if cell == None:
            logger.info("Word: %s not found in fasttext" % (cell))
        return cell

    def add_bert(self, predicted: str, score, tokenizer: BertTokenizer):
        self._bert[predicted] = score
        tokenized = tokenizer.tokenize(predicted)
        normal_form = PredictionsForWord.get_normal_form(tokenized[0])
        self._normalized_bert[normal_form] = predicted

    def add_skipgram(self, predicted: str, score):
        self._skipgram[predicted] = score

    def add_fasttext(self, predicted: str, score):
        self._fasttext[predicted] = score

    def get_normal_form(word: str):
        morph_parse = PredictionsForWord._morph.parse(word)
        return morph_parse[0].normal_form

    def find_bert(self, tokenizer: BertTokenizer):
        logger = logging.getLogger("graduateWork.dictionary_top.find_bert")
        for dictionary in PredictionsForWord._vocabulary:
            for dict_word in dictionary._dictionary:
                tokenized = tokenizer.tokenize(dict_word)
                if tokenized.__len__() > 1:
                    logger.info(
                        "Word: %s have more then one token. Tokens count: %d" % (dict_word, tokenized.__len__()))
                normalized_token = PredictionsForWord.get_normal_form(tokenized[0])
                needed_bert_form = self.get_normalized_bert(normalized_token)
                if needed_bert_form == None:
                    logger.info("Word: %s doesnt have score in bert model" % (tokenized[0]))
                    continue
                score = self.get_bert(needed_bert_form)
                if score == None:
                    logger.info("Word: %s doesnt have score in bert model" % (tokenized[0]))
                    continue
                self._bert_dictionary[needed_bert_form] = (score, dictionary._name)

    def find_skipgram(self, tokenizer: udpipe.Pipeline):
        logger = logging.getLogger("graduateWork.dictionary_top.find_skipgram")
        for dictionary in PredictionsForWord._vocabulary:
            for dict_word in dictionary._dictionary:
                unified = udpipe.unify_sym(dict_word)
                tokenized = udpipe.process(tokenizer, text=unified, keep_punct=True)
                if tokenized == None:
                    logger.info("Word: %s doesnt have token in skipgram model" % (dict_word))
                    continue
                score = self.get_skipgram(tokenized[0])
                if score == None:
                    logger.info("Word: %s doesnt have score in skipgram model" % (tokenized[0]))
                    continue
                self._skipgram_dictionary[dict_word] = (score, dictionary._name)

    def find_fasttext(self):
        logger = logging.getLogger("graduateWork.dictionary_top.find_fasttext")
        for dictionary in PredictionsForWord._vocabulary:
            for dict_word in dictionary._dictionary:
                unified = udpipe.unify_sym(dict_word)
                score = self.get_fasttext(unified)
                if score == None:
                    logger.info("Word: %s doesnt have score in fasttext model" % (unified))
                    continue
                self._fasttext_dictionary[dict_word] = (score, dictionary._name)

    def find_all(self, bert_tokenizer: BertTokenizer, udpipe: udpipe.Pipeline):
        self.find_bert(bert_tokenizer)
        self.find_skipgram(udpipe)
        self.find_fasttext()

    def add_vocabulary(path: str):
        dict = Dictionary(path)
        PredictionsForWord._vocabulary.append(dict)

    def sinonimize(self):
        tmp = {}
        for candidates in self._bert_dictionary:
            dict_name = self._bert_dictionary[candidates][1]
            value = tmp.get(dict_name, 0)
            tmp[dict_name] = value + 1
        max = 0
        leader = None
        for candidates in tmp:
            if tmp[candidates] > max:
                max = tmp[candidates]
                leader = candidates
        if leader == None:
            return self._word
        word = []
        scores = np.array([])
        for predicted_word in self._bert_dictionary:
            cell = self._bert_dictionary[predicted_word]
            if cell[1] == leader:
                word.append(predicted_word)
                scores = np.append(scores, [cell[0]])

        normalized = softmax(scores)
        print(word)
        print(normalized)
        predicted = np.random.choice(word, 1, p=normalized)
        print(predicted)
        return predicted[0]

    def sinonimize_skipg(self):
        tmp = {}
        for candidates in self._skipgram_dictionary:
            dict_name = self._skipgram_dictionary[candidates][1]
            value = tmp.get(dict_name, 0)
            tmp[dict_name] = value + 1
        max = 0
        leader = None
        for candidates in tmp:
            if tmp[candidates] > max:
                max = tmp[candidates]
                leader = candidates
        if leader == None:
            return self._word
        word = []
        scores = np.array([])
        for predicted_word in self._skipgram_dictionary:
            cell = self._skipgram_dictionary[predicted_word]
            if cell[1] == leader:
                word.append(predicted_word)
                scores = np.append(scores, [cell[0]])

        normalized = softmax(scores)
        print(word)
        print(normalized)
        predicted = np.random.choice(word, 1, p=normalized)
        print(predicted)
        return predicted[0]

    def sinonimize_fast(self):
        tmp = {}
        for candidates in self._fasttext_dictionary:
            dict_name = self._fasttext_dictionary[candidates][1]
            value = tmp.get(dict_name, 0)
            tmp[dict_name] = value + 1
        max = 0
        leader = None
        for candidates in tmp:
            if tmp[candidates] > max:
                max = tmp[candidates]
                leader = candidates
        if leader == None:
            return self._word
        word = []
        scores = np.array([])
        for predicted_word in self._fasttext_dictionary:
            cell = self._fasttext_dictionary[predicted_word]
            if cell[1] == leader:
                word.append(predicted_word)
                scores = np.append(scores, [cell[0]])

        normalized = softmax(scores)
        print(word)
        print(normalized)
        predicted = np.random.choice(word, 1, p=normalized)
        print(predicted)
        return predicted[0]



class PredictionsForText:

    def __init__(self, text: str):
        """Constructor"""
        self._text = text
        self._words = {}
        nlp = spacy.load("ru2")
        doc = nlp(text)
        for token in doc:
            self._words[token.text] = PredictionsForWord(token.text)

    def __str__(self):
        stroka = "------------------------------------------"
        stroka += "Предсказания для текста: " + self._text + "\n"
        for word in self._words:
            stroka += str(word) + "\n"
        stroka += "------------------------------------------"
        return stroka

    def add_bert(self, word: str, predicted: str, score, tokenizer: BertTokenizer):
        logger = logging.getLogger("graduateWork.PredictionsForText.add_bert")
        predictions = self._words.get(word)
        if predictions == None:
            logger.info("Word: %s not found in current text" % (word))
            return
        predictions.add_bert(predicted, score, tokenizer)

    def add_skipgram(self, word: str, predicted: str, score):
        logger = logging.getLogger("graduateWork.PredictionsForText.add_skipgram")
        predictions = self._words.get(word)
        if predictions == None:
            logger.info("Word: %s not found in current text" % (word))
            return
        predictions.add_skipgram(predicted, score)

    def add_fasttext(self, word: str, predicted: str, score):
        logger = logging.getLogger("graduateWork.PredictionsForText.add_fasttext")
        predictions = self._words.get(word)
        if predictions == None:
            logger.info("Word: %s not found in current text" % (word))
            return
        predictions.add_fasttext(predicted, score)

    def find_all(self, bert_tokenizer: BertTokenizer, udpipe: udpipe.Pipeline):
        for word in self._words:
            self._words[word].find_all(bert_tokenizer, udpipe)

    def sinonimize(self):
        new_sentense = []
        for words in self._words:
            new_sentense.append(self._words[words].sinonimize())
        return ' '.join(new_sentense)

    def sinonimize_skipgram(self):
        new_sentense = []
        for words in self._words:
            new_sentense.append(self._words[words].sinonimize_skipg())
        return ' '.join(new_sentense)

    def sinonimize_fasttext(self):
        new_sentense = []
        for words in self._words:
            new_sentense.append(self._words[words].sinonimize_fast())
        return ' '.join(new_sentense)

