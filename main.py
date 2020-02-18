import torch
from transformers import AutoTokenizer, AutoModel
import logging
from pandas import DataFrame
logging.basicConfig(level=logging.INFO)

modelpath = "DrMatters/rubert_cased"
tokenizer = AutoTokenizer.from_pretrained(modelpath)

text = "Британский дизайнер и предприниматель, внук королевы Великобритании Елизаветы II Дэвид Армстронг-Джонс и его супруга Серена решили подать на развод"


def my_tokenizer(str):
    dictionary = dict()
    tokenized_text = list(())
    for word in str.split():
        # if word == "[CLS]" or word == "[SEP]":
        #     continue
        tokenized_word = tokenizer.tokenize(word)
        tokenized_text.extend(tokenized_word)
        dictionary[word] = tokenized_word
    return (dictionary, tokenized_text)


my_tokenized_text = my_tokenizer(text)

model = AutoModel.from_pretrained(modelpath)
model.eval()

final_dict = dict()

for subword_index, subword in enumerate(my_tokenized_text[1]):
    my_tokenized_text[1][subword_index] = '[MASK]'
    indexed_tokens = tokenizer.convert_tokens_to_ids(my_tokenized_text[1])

    # segments_ids = [1] * len(my_tokenized_text[1])
    # segments_ids[0] = 0
    # segments_ids[1] = 0
    # segments_ids[2] = 0
    # segments_ids[3] = 0
    # segments_ids[4] = 0

    tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])

    predictions = model(tokens_tensor)[0]
    top_100 = list(())
    for value, ids in zip(*torch.topk(predictions[0, subword_index], 100, largest=True)):
        predicted_token = tokenizer.decode([ids.item()])
        top_100.append((predicted_token, value.item()))
    final_dict[subword] = top_100
    my_tokenized_text[1][subword_index] = subword

print("Input text:", text)
print(my_tokenized_text[0])
print(my_tokenized_text[1])

export_dict = dict()

for key, value in final_dict.items():
    print("-----Subword:", key, "------")
    export_dict[key] = value
    for (predicted_token, value) in value:
        print("Predicated token:", predicted_token, "with value:", value)

excel_frame = DataFrame(export_dict)
excel_frame.to_excel('output.xlsx', sheet_name='Sheet1')


