import torch
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, AutoModelWithLMHead
import logging
import openpyxl
import pymorphy2
import gensim
from pandas import DataFrame
from spacy.lang.ru import Russian


logging.basicConfig(level=logging.INFO)

modelpath = "DrMatters/rubert_cased"
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = AutoModelWithLMHead.from_pretrained(modelpath)
morph = pymorphy2.MorphAnalyzer()
model.eval()

text = "он подарил мне блестящий железный меч"

print(tokenizer.encode(text))

# def my_tokenizer(*str):
#     dictionary = dict()
#     tokenized_text = list(())
#     for word in str.split():
#         tokenized_word = tokenizer.tokenize(word)
#         tokenized_text.extend(tokenized_word)
#         dictionary[word] = tokenized_word
#     return (dictionary, tokenized_text)

splitted_text = list(())

nlp = Russian()
doc = nlp(text)
for token in doc:
    splitted_text.append(token.text)
final_dict = dict()

for word_index, word in enumerate(splitted_text):
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
    for value, ids in zip(*torch.topk(predictions[0, masked_index], 100, largest=True)):
        predicted_token = tokenizer.convert_ids_to_tokens([ids.item()])
        top_100.append((predicted_token, value.item()))
    final_dict[word] = top_100
    splitted_text[word_index] = word

taiga_fasttext = gensim.models.KeyedVectors.load("taiga_fasttext/model.model")
taiga_skipgram = gensim.models.KeyedVectors.load_word2vec_format("taiga_skipgram/model.bin", binary=True)

export_dict_fasttext = dict()
export_dict_skipgram = dict()

for word in splitted_text:
    # есть ли слово в модели? Может быть, и нет
    p = morph.parse(word)[0]
    preproced_word = word + "_" + str(p.tag).split(',')[0]
    if preproced_word in taiga_fasttext:
        print(preproced_word)
        # выдаем 10 ближайших соседей слова:
        for i in taiga_fasttext.most_similar(positive=[preproced_word], topn=100):
            print('taiga_fasttext \n')
            # слово + коэффициент косинусной близости
            print(i[0], i[1])
            export_dict_fasttext[preproced_word] = i
        print('\n')
    else:
        # Увы!
        print(preproced_word + ' is not present in the taiga_fasttext model')
        export_dict_fasttext[preproced_word] = "NOT FOUND"
    if preproced_word in taiga_skipgram:
        print(preproced_word)
        # выдаем 10 ближайших соседей слова:
        for i in taiga_skipgram.most_similar(positive=[preproced_word], topn=100):
            print('taiga_skipgram \n')
            # слово + коэффициент косинусной близости
            print(i[0], i[1])
            export_dict_skipgram[preproced_word] = i
        print('\n')
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

excel_frame = DataFrame(export_dict)
excel_frame.to_excel('output.xlsx', sheet_name='Bert')
excel_frame = DataFrame(export_dict_fasttext)
excel_frame.to_excel('output.xlsx', sheet_name='FastText')
excel_frame = DataFrame(export_dict_skipgram)
excel_frame.to_excel('output.xlsx', sheet_name='Skipgram')
