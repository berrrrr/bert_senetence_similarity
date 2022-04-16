[![Version](https://img.shields.io/badge/version-v1.0.0--SNAPSHOT-1abc9c.svg)](README.md)

# # bert_senetence_similarity
+ bert를 이용해 입력받은 문장의 동일 여부를 판단하는 모델

## 기술스택
+ flask
+ pytorch
+ BertForSequenceClassification

## 사용 데이터셋 
https://github.com/kakaobrain/KorNLUDatasets/tree/master/KorSTS

## 호출샘플1
### request
```
POST http://localhost:5000/predict
Accept: application/json
Content-Type: application/json

{
  "sentence_1": "한 남자가 양파를 자르고 있다.",
  "sentence_2": "한 남자가 양파를 잘랐다."
}
```
### response
```json
{
  "label": "1",
  "label_kor": "일치",
  "logits": [
    [
      -3.6315908432006836,
      3.9494943618774414
    ]
  ],
  "time": "0.04401063919067383",
  "version": "20200304"
}

```
## 호출샘플2
### request
```
POST http://localhost:5000/predict
Accept: application/json
Content-Type: application/json

{
  "sentence_1": "한 남자가 파스타를 먹고 있다.",
  "sentence_2": "한 남자가 기타를 치고 있다."
}
```
### response
```json
{
  "label": "0",
  "label_kor": "불일치",
  "logits": [
    [
      4.073559761047363,
      -4.510875701904297
    ]
  ],
  "time": "0.047009944915771484",
  "version": "20200304"
}
```