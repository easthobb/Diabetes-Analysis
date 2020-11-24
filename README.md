# Diabetes-Analysis
20-2 개별연구 다양한 관점에서의 보건영양데이터 분석

본 레포는 '다양한 관점에서의 보건영양데이터'를 주제로, 제7기 2018 국민 보건영양조사의 데이터를 이용합니다. 본 연구에서의 핵심 목표는 다음과 같습니다. 

 ## Objects
 **Key Object : 당뇨병 예측모델 개발**  
 
 **sub Objects :**

-  국민보건영양데이터 2018 중 당뇨병 관련 유력 변수 선별

-  당뇨병 환자군/비환자군 , 노인/비노인(65세 기준)의 데이터 구조 시각화

-  데이터에 대해 ML적 기법을 적용시켜 SVM/Catboost/RF/GLM 등 다양한 분류기를 학습

- 학습된 분류기의 정확성을 전체 정확도, 당뇨병 판펼도(2형오류/정상판별)로 판단
- 가장 최적화된 성능을 가진 모델을 선별해 개별 환자 데이터를 입력해 유병 여부를 판단할 수 있는 코드 개발


## Files

- dEDA.py : 연구 전반적인 코드 수행(변수 drop , 스케일링, 분류 모델 train / fit 등 model 성능 판단) 

- dEDA.ipynb : dEDA.py block by block 수행(graph)

- decision.py : json 형태 샘플 데이터에 대해서 분류(당뇨병/비당뇨병) 수행 

- MissingValue_KNN.ipynb : 결측치 처리(범주형) 수행코드

- MissingValue_Median.ipynb : 결측치 처리(연속형) 수행코드

- PCA_EDA.ipynb : PCA 테스팅 (비도입 결정)

- 2018_SEL_FILLED.csv : 가공된 2018 국민보건영양데이터 


## 현재 진행 상황
- 유력변수 선별 완료(11.24) // VIF , Corelation

- 데이터 구조 시각화 // Histogram pairplot HC

- 분류기 사용 및 train 완료 // 현재 SVM 만

## 결과
(노션에서 업로드)

##  Keyword
ML, SVM, CATBOOST,XGBOOST, VIF, Corelation, diabetes, Classifier, 

## 도전과제

- Decision.py 모델로 사용자에게 선별 변수를 입력받아 당뇨병 판별을 해주는 Web app 개발