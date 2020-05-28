from transformers import BertTokenizer
import logging
import udpipe
import spacy

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

    def __init__(self, word: str):
        """Constructor"""
        self._word = word
        self._bert = {}
        self._skipgram = {}
        self._fasttext = {}
        self._bert_dictionary = {}
        self._skipgram_dictionary = {}
        self._fasttext_dictionary = {}

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

    def add_bert(self, predicted: str, score):
        self._bert[predicted] = score

    def add_skipgram(self, predicted: str, score):
        self._skipgram[predicted] = score

    def add_fasttext(self, predicted: str, score):
        self._fasttext[predicted] = score

    def find_bert(self, tokenizer: BertTokenizer):
        logger = logging.getLogger("graduateWork.dictionary_top.find_bert")
        for dictionary in PredictionsForWord._vocabulary:
            for dict_word in dictionary._dictionary:
                tokenized = tokenizer.tokenize(dict_word)
                if tokenized.__len__() > 1:
                    logger.info(
                        "Word: %s have more then one token. Tokens count: %d" % (dict_word, tokenized.__len__()))
                score = self.get_bert(tokenized[0])
                if score == None:
                    logger.info("Word: %s doesnt have score in bert model" % (tokenized[0]))
                    continue
                self._bert_dictionary[dict_word] = (score, dictionary._name)

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

    def add_bert(self, word: str, predicted: str, score):
        logger = logging.getLogger("graduateWork.PredictionsForText.add_bert")
        predictions = self._words.get(word)
        if predictions == None:
            logger.info("Word: %s not found in current text" % (word))
            return
        predictions.add_bert(predicted, score)

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
