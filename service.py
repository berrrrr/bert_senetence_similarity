import logging
import time

import numpy as np
import torch
from flask import Flask
from transformers import BertForSequenceClassification
from transformers import BertTokenizer

# 로깅
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 플라스크앱
app = Flask(__name__)

# BERT model, tokenizer
# bert-base-multilingual-cased 를 사용
# https://github.com/google-research/bert/blob/master/multilingual.md
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


# 추가로 학습한 모델이 있다면 아래와 같이 로드
PATH = './model/model.pth'
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

# 모델 평가모드로 전환
model.eval()

def convert_input_data(data):
    """
    단어1, 단어2를 입력받아 bert에 입력할 수 있는 embedding 형태로 변환한다

    :param data:
    :return:
    """
    max_sequence_length = 64
    max_bert_input_length = 0

    for sentence_pair in data:
        sentence_1_tokenized, sentence_2_tokenized = tokenizer.tokenize(
            sentence_pair['sentence_1']), tokenizer.tokenize(sentence_pair['sentence_2'])

        max_bert_input_length = max(max_bert_input_length, len(sentence_1_tokenized) + len(sentence_2_tokenized) + 3)
        sentence_pair['sentence_1_tokenized'] = sentence_1_tokenized
        sentence_pair['sentence_2_tokenized'] = sentence_2_tokenized

    dataset_input_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_token_type_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_attention_masks = torch.empty((len(data), max_bert_input_length), dtype=torch.long)

    for idx, sentence_pair in enumerate(data):
        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in sentence_pair['sentence_1_tokenized']:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        for token in sentence_pair['sentence_2_tokenized']:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_masks = [1] * len(input_ids)
        while len(input_ids) < max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)
            input_type_ids.append(0)

        dataset_input_ids[idx] = torch.tensor(input_ids, dtype=torch.long)
        dataset_token_type_ids[idx] = torch.tensor(input_type_ids, dtype=torch.long)
        dataset_attention_masks[idx] = torch.tensor(attention_masks, dtype=torch.long)

    return dataset_input_ids, dataset_token_type_ids, dataset_attention_masks


def predict(received_data):
    """
    입력받은 데이터를 모델에 넣어 예측값을 뽑아낸다

    :param received_data:
    :return:
    """
    start = time.time()

    data = []
    data.append({
        'sentence_1': received_data['sentence_1'],
        'sentence_2': received_data['sentence_2'],
    })

    inputs, types, masks = convert_input_data(data)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(inputs,
                        token_type_ids=types,
                        attention_mask=masks)

    logits = outputs[0].numpy().tolist()
    label = str(np.argmax(logits))
    label_kor = str((lambda x: '일치' if x == 1 else '불일치')(np.argmax(logits)))

    end = time.time()

    result = {'label_kor': label_kor, 'label': label, 'logits': logits, 'version': '20200304', 'time': str(end - start)}
    return result
