import torch
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, AutoModelWithLMHead
import logging
import openpyxl
import gensim
from pandas import DataFrame, ExcelWriter
import spacy
import pymorphy2
import udpipe
from datetime import datetime

from dictionary_top import Dictionary, PredictionsForWord, PredictionsForText

logging.config.fileConfig('log_config.conf')
logger = logging.getLogger("graduateWork")

logger.info("Program started")

dictionaries = ["cost.txt", "material.txt", "positive.txt", "usability.txt"]
for path in dictionaries:
    PredictionsForWord.add_vocabulary(path)

modelpath = "DrMatters/rubert_cased"
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = AutoModelWithLMHead.from_pretrained(modelpath)
model.eval()

text = "он подарил мне блестящий меч"
my_predictions = PredictionsForText(text)
topn = 30
print(tokenizer.encode(text))


splitted_text = list(())

# https://github.com/buriy/spacy-ru
nlp = spacy.load("ru2")
doc = nlp(text)
splitted_text.append("[CLS]")
for token in doc:
    splitted_text.append(token.text)
# splitted_text.append("[SEP]")
final_dict = dict()

for word_index, word in enumerate(splitted_text):
    if word == "[SEP]" or word == "[CLS]":
        continue
    splitted_text[word_index] = '[MASK]'

    masked_index = None
    tokenized_text = list(())

    for word2 in splitted_text:
        tokenized_word = tokenizer.tokenize(word2)
        tokenized_text.extend(tokenized_word)
        if word2 == '[MASK]':
            masked_index = len(tokenized_text) - 1

    print(masked_index)
    print(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    print(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])

    predictions = model(tokens_tensor)[0]
    top_100 = list(())
    for value, ids in zip(*torch.topk(predictions[0, masked_index], topn, largest=True)):
        predicted_token = tokenizer.convert_ids_to_tokens([ids.item()])
        top_100.append((predicted_token, value.item()))
        print(predicted_token)
        my_predictions.add_bert(word, predicted_token[0], value.item(), tokenizer)
    final_dict[word] = top_100
    splitted_text[word_index] = word

taiga_fasttext = gensim.models.KeyedVectors.load("taiga_fasttext/model.model")
taiga_skipgram = gensim.models.KeyedVectors.load_word2vec_format("taiga_skipgram/model.bin", binary=True)

export_dict_fasttext = dict()
export_dict_skipgram = dict()

model = udpipe.init()
process_pipeline = udpipe.Pipeline(model, 'tokenize', udpipe.Pipeline.DEFAULT, udpipe.Pipeline.DEFAULT, 'conllu')
for word in doc:
    if str(word) == "[SEP]" or str(word) == "[CLS]":
        continue
    # есть ли слово в модели? Может быть, и нет
    res = udpipe.unify_sym(str(word))
    preproced_word = udpipe.process(process_pipeline, text=res, keep_punct=True)[0]
    if word in taiga_fasttext:
        print(word)
        top_100 = list(())
        print('taiga_fasttext \n')
        # выдаем 10 ближайших соседей слова:
        for i in taiga_fasttext.most_similar(positive=[str(word)], topn=topn):
            my_predictions.add_fasttext(str(word), i[0], i[1])
            top_100.append((i[0], i[1]))
        export_dict_fasttext[word] = top_100
    else:
        # Увы!
        print(word + ' is not present in the taiga_fasttext model')
        export_dict_fasttext[word] = "NOT FOUND"
    if preproced_word in taiga_skipgram:
        print(preproced_word)
        top_100 = list(())
        print('taiga_skipgram \n')
        # выдаем 10 ближайших соседей слова:
        for i in taiga_skipgram.most_similar(positive=[preproced_word], topn=topn):
            my_predictions.add_skipgram(str(word), i[0], i[1])
            top_100.append((i[0], i[1]))
        export_dict_skipgram[preproced_word] = top_100
    else:
        # Увы!
        print(preproced_word + ' is not present in the taiga_skipgram model')
        export_dict_skipgram[preproced_word] = "NOT FOUND"

print("Input text:", text)

export_dict = dict()

for key, value in final_dict.items():
    print("-----Subword:", key, "------")
    export_dict[key] = value
    for (predicted_token, value) in value:
        print("Predicated token:", predicted_token, "with value:", value)

writer = ExcelWriter(str(datetime.now().isoformat()) + 'output.xlsx', engine='xlsxwriter')
DataFrame(export_dict).to_excel(writer, sheet_name='Bert')
DataFrame(export_dict_fasttext).to_excel(writer, sheet_name='FastText')
DataFrame(export_dict_skipgram).to_excel(writer, sheet_name='Skipgram')
writer.save()
my_predictions.find_all(tokenizer, process_pipeline)
print("---------BRET----------")
print(my_predictions.sinonimize())
print(my_predictions.sinonimize())
print(my_predictions.sinonimize())
print(my_predictions.sinonimize())
print("---------SKIPGRAM----------")
print(my_predictions.sinonimize_skipgram())
print(my_predictions.sinonimize_skipgram())
print(my_predictions.sinonimize_skipgram())
print(my_predictions.sinonimize_skipgram())
print("----------FASTTEXT---------")
print(my_predictions.sinonimize_fasttext())
print(my_predictions.sinonimize_fasttext())
print(my_predictions.sinonimize_fasttext())
print(my_predictions.sinonimize_fasttext())
logger.info("------------------BEFORE PRING--------------")
print(my_predictions)
logger.info("Done!")
