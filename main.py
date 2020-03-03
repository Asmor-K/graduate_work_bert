import torch
from transformers import AutoTokenizer, BertTokenizer, BertForMaskedLM, AutoModelWithLMHead
import logging
from pandas import DataFrame
logging.basicConfig(level=logging.INFO)

modelpath = "DrMatters/rubert_cased"
tokenizer = BertTokenizer.from_pretrained(modelpath)
model = AutoModelWithLMHead.from_pretrained(modelpath)

model.eval()

text = "Развеиваем завесу мистики над управлением памятью в программном обеспечении и подробно рассматриваем возможности, предоставляемые современными языками программирования"

print(tokenizer.encode(text))


def my_tokenizer(*str):
    dictionary = dict()
    tokenized_text = list(())
    for word in str.split():
        tokenized_word = tokenizer.tokenize(word)
        tokenized_text.extend(tokenized_word)
        dictionary[word] = tokenized_word
    return (dictionary, tokenized_text)

splitted_text = text.split()

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

print("Input text:", text)

export_dict = dict()

for key, value in final_dict.items():
    print("-----Subword:", key, "------")
    export_dict[key] = value
    for (predicted_token, value) in value:
        print("Predicated token:", predicted_token, "with value:", value)

excel_frame = DataFrame(export_dict)
excel_frame.to_excel('output.xlsx', sheet_name='Sheet1')


