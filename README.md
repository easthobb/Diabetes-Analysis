# Diabetes-Analysis
21년 한국통신학회 동계 학술종합발표회 투고 논문(17E-54) "XGBoost 기반 당뇨병 예측 알고리즘 연구:국민건강영양조사 2016~2018을 이용하여"의 Open Source Repository 입니다.(ISSN:2383-8302(Online) Vol.74)본 레포는 '다양한 관점에서의 보건영양데이터'를 주제로, 제7기 2018 국민 보건영양조사의 데이터를 이용했습니다.본 연구에서의 핵심 목표는 **당뇨병 예측모델 개발** 이며 본 Repository에서는 프로젝트의 아키텍쳐, 코드, 결과 그래프 등이 포함되어 있습니다.연구 전반의 기록과 과정을 노션 페이지에서도 확인할 수 있습니다.

https://www.notion.so/hobbeskim/XGBoost-20-2-2740a0d75839481b8cbefa7cdab69466
## Architecture
![Archi](https://user-images.githubusercontent.com/57410044/106548637-b39bfc00-6552-11eb-91ce-3b629b599dfc.png)

## Model Evaluation
![roc](https://user-images.githubusercontent.com/57410044/106548802-070e4a00-6553-11eb-92ba-c49ce1859fd2.jpg)
![values](https://user-images.githubusercontent.com/57410044/106548809-0a093a80-6553-11eb-9e4c-09cdf0572662.png)


## Files Description

- ./Data : 실제로 연구에 이용한 데이터 파일, 국민건강영양조사 데이터 가공
- ./Data EDA : 데이터를 분석하고 시각화 한 과정의 자료가 존재
- ./Data Processing : 모델 학습과 예측을 위해 전처리를 수행한 과정의 파일 존재 
- ./Etc : 기타 개발과정의 오류나 산출물 등
- ./Results : 결과 그래프 원본 이미지
- FOLD_EVALUATION.ipynb : 가공된 데이터 -> 최종 결과 수행 시퀀스


## 향후 도전과제
- 시계열 추적 연구에 기반한 보다 사전적인 예측이 가능한 Data Set과 모델이 필요
- **선별된 변수와 훈련된 모델로 사용자에게 입력을 받아 당뇨병 유병 여부, 확률 등을 예측하는 Web App 개발**
- 당뇨병의 유병 여부 및 사전 진단에 사용 가능한 생체 지표 발굴 중요성 시사

##  Keyword
ML,XGBOOST, VIF, Corelation, diabetes, Classifier, 당뇨병 예측, 국민건강영양조사, 알고리즘

## Acknowledgement
- 본 연구는 과학기술정보통신부 및 정보통신기획평가원의 대학ICT연구센터지원사업의 연구 결과로 수행되었음 (IITP-2021- 2020-0-01789)
- 본 연구는 과학기술정보통신부 및 정보통신기획평가원의 SW중심대학지원사업의 연구결과로 수행되었음(2016-0-00017)