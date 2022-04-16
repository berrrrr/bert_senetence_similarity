import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
from transformers import BertForSequenceClassification, AdamW, BertConfig
import numpy as np
import datetime

import time

import random
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

train_full = pd.read_csv('data/sts-train.tsv', delimiter='\t', error_bad_lines=False)
columns = ['sentence1', 'sentence2', 'score']
print(train_full.shape)
train = train_full.loc[:, columns]
train = train.fillna(value="")

"""
# **전처리 - 훈련셋**
"""

# 문장 추출
sentences_1 = train['sentence1']
sentences_2 = train['sentence2']

# 라벨 추출
scores = train['score']
train['label'] = scores.apply(lambda x: 1 if x > 3.0 else 0)
labels = train['label'].values

data = []
for idx, (sentence_1, sentence_2, label) in enumerate(zip(sentences_1, sentences_2, labels)):
    data.append({
        'index': idx,
        'sentence_1': sentence_1,
        'sentence_2': sentence_2,
        'similarity': label
    })

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


# 전처리 함수정의

def bert_sentence_pair_preprocessing(data: list, tokenizer: BertTokenizer):
    max_bert_input_length = 128
    for sentence_pair in data:
        sentence_1_tokenized, sentence_2_tokenized = tokenizer.tokenize(
            sentence_pair['sentence_1']), tokenizer.tokenize(sentence_pair['sentence_2'])

        max_bert_input_length = max(max_bert_input_length, len(sentence_1_tokenized) + len(sentence_2_tokenized) + 3)
        sentence_pair['sentence_1_tokenized'] = sentence_1_tokenized
        sentence_pair['sentence_2_tokenized'] = sentence_2_tokenized

    dataset_input_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_token_type_ids = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_attention_masks = torch.empty((len(data), max_bert_input_length), dtype=torch.long)
    dataset_labels = torch.empty((len(data), 1), dtype=torch.long)

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
        if 'similarity' not in sentence_pair or sentence_pair['similarity'] is None:
            dataset_labels[idx] = torch.tensor(float('nan'), dtype=torch.long)
        else:
            dataset_labels[idx] = torch.tensor(sentence_pair['similarity'], dtype=torch.long)

    # BERT의 maximum embedding dim을 넘지 않도록 512로 잘라줌
    return dataset_input_ids[:, :512], dataset_token_type_ids[:, :512], dataset_attention_masks[:, :512], dataset_labels


input_ids_eval, token_type_ids_eval, attention_masks_eval, correct_labels_eval = bert_sentence_pair_preprocessing(data,
                                                                                                                  tokenizer)

# 훈련셋과 검증셋으로 분리
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids_eval,
                                                                                    correct_labels_eval,
                                                                                    random_state=2018,
                                                                                    test_size=0.1)

# 어텐션 마스크를 훈련셋과 검증셋으로 분리
train_masks, validation_masks, _, _ = train_test_split(attention_masks_eval,
                                                       input_ids_eval,
                                                       random_state=2018,
                                                       test_size=0.1)

# 타입인덱스를 훈련셋과 검증셋으로 분리
train_types, validation_types, _, _ = train_test_split(token_type_ids_eval,
                                                       input_ids_eval,
                                                       random_state=2018,
                                                       test_size=0.1)

# 데이터를 파이토치의 텐서로 변환
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_types = torch.tensor(train_types)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
validation_types = torch.tensor(validation_types)

# 배치 사이즈
batch_size = 4

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# 파이토치의 DataLoader로 입력, 마스크, 라벨을 묶어 데이터 설정
# 학습시 배치 사이즈 만큼 데이터를 가져옴
train_data = TensorDataset(train_inputs, train_masks, train_types, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_types, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# 분류를 위한 BERT 모델 생성
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

# 디바이스 설정
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.cuda()
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using the CPU instead.')


# 옵티마이저 설정
optimizer = AdamW(model.parameters(),
                  lr=2e-5,  # 학습률
                  eps=1e-8  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                  )
# 에폭수
epochs = 5

# 총 훈련 스텝 : 배치반복 횟수 * 에폭
total_steps = len(train_dataloader) * epochs
from transformers import get_linear_schedule_with_warmup

# 학습률을 조금씩 감소시키는 스케줄러 생성
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# confusion matrix
def confusion(prediction, truth):
    pred_flat = np.argmax(prediction, axis=1).flatten()
    truth_flat = truth.flatten()

    confusion_vector = pred_flat / truth_flat

    true_positives = np.sum(confusion_vector == 1).item()
    false_positives = np.sum(confusion_vector == float('inf')).item()
    true_negatives = np.sum(np.isnan(confusion_vector)).item()
    false_negatives = np.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))


# 재현을 위해 랜덤시드 고정
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 그래디언트 초기화
model.zero_grad()
# 에폭만큼 반복
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # 시작 시간 설정
    t0 = time.time()

    # 로스 초기화
    total_loss = 0

    # 훈련모드로 변경
    model.train()

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(train_dataloader):
        # 경과 정보 표시
        if step % 500 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_input_types, b_labels = batch

        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=b_input_types,
                        attention_mask=b_input_mask,
                        labels=b_labels)

        # 로스 구함
        loss = outputs[0]

        # 총 로스 계산
        total_loss += loss.item()

        # Backward 수행으로 그래디언트 계산
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 그래디언트를 통해 가중치 파라미터 업데이트
        optimizer.step()

        # 스케줄러로 학습률 감소
        scheduler.step()

        # 그래디언트 초기화
        model.zero_grad()

    # 평균 로스 계산
    avg_train_loss = total_loss / len(train_dataloader)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    # 시작 시간 설정
    t0 = time.time()

    # 평가모드로 변경
    model.eval()

    # 변수 초기화
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0

    # 데이터로더에서 배치만큼 반복하여 가져옴
    for batch in validation_dataloader:
        # 배치를 GPU에 넣음
        batch = tuple(t.to(device) for t in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask, b_input_types, b_labels = batch

        # 그래디언트 계산 안함
        with torch.no_grad():
            # Forward 수행
            outputs = model(b_input_ids,
                            token_type_ids=b_input_types,
                            attention_mask=b_input_mask)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # 출력 로짓과 라벨을 비교하여 정확도 계산
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        # confusion matrix 계
        tmp_tp, tmp_fp, tmp_tn, tmp_fn = confusion(logits, label_ids)
        tp += tmp_tp
        fp += tmp_fp
        tn += tmp_tn
        fn += tmp_fn

        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("tp: ", tp)
    print("fp: ", fp)
    print("tn: ", tn)
    print("fn: ", fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("precesion : ", precision)
    print("recall : ", recall)
    print("f1 : ", f1)

print("")
print("Training complete!")

# 학습한 모델 저장
PATH = "./model/model.pth"
torch.save(model.state_dict(), PATH)
