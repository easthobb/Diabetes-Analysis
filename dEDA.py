# STEP 1 - 의존성 로드 - O

# STEP 2 - 데이터 임포팅 - O

# STEP 3 - KNN 으로 missing value filling 
# -> O (민경 코드)

# STEP 4 - Data 자체의 구조와 분석 
# -> (histogram-O, pairplot-O, HC-진행중)

# STEP 5 = 변수간 상관관계 및 다중공선성(VIF) 분석 
#-> (상관관계-O,다중공선성-일부변수에 대해서만 수행->진행) 

# STEP 6 - 변수의 RF 모델에서 나온 중요도 산출 
#-> O // 날리고 살릴 기준이 필요

# STEP 7 - step5, step 6의 결론을 토대로 변수 drop 
#-> 최종 관심 변수 선택 - 

# STEP 8 - 파이프라이닝 - 스케일링 / 모델링 결정 
#-> 모델별로 달라서 어렵다

# STEP 9 - 피팅한 모델들의 성능 비교 후 
#-> 현재 2형 오류가 너무 많음 

# STEP 10 - 최종 관심 변수가 입력된 단일 샘플의 당뇨병 예측 
#-> 코드 구현 필요 




# STEP 1 -  의존성 로드 ############################################################
import numpy as np
import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns # seaborn ref : https://greeksharifa.github.io/machine_learning/2019/12/05/Seaborn-Module/
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import statsmodels.api as sm;
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
%matplotlib inline

# STEP 2 - 데이터 임포팅 ############################################################

df = pd.read_csv("2018_sel_KNN.csv") ## 파일리딩!
df.info()

# STEP 3 - KNN 으로 missing value filling ############################################################



# STEP 4 - Data에 대한 구조와 분석

df.hist(figsize=(20,20))#histogram plotting
sns.paiplot(df)

# STEP 5 = 변수간 상관관계 및 다중공선성(VIF) 분석 #################################################################### 
# ref : https://bkshin.tistory.com/entry/DATA-20-%EB%8B%A4%EC%A4%91%EA%B3%B5%EC%84%A0%EC%84%B1%EA%B3%BC-VIF */

## 변수간 상관관계 분석
t=df.corr(method='pearson') ## de1_pr 현재 당뇨병 유병 여부와 상관관계가 있는 변수들
t.DE1_pr ## 각 변수들간의 pearson 상관계수 정렬 p>0.1  -> age, ho_incm, cfam, genertnm 
sns.heatmap(df.corr(),cmap="YlGnBu")

## VIF 산출
df['intercept'] = 1
lm = sm.OLS(df['DE1_pr'], df)
results = lm.fit()
results.summary()
sns.pairplot(df[['DE1_pr','age', 'wt_ntr','D_1_1','Total_slp_wk','HE_ht','HE_wc','HE_obe']])
y, X = dmatrices('DE1_pr ~ age + wt_ntr + D_1_1 + Total_slp_wk + HE_ht + HE_wc + HE_obe' , df, return_type = 'dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
vif


# STEP 6 - 변수의 RF 모델에서 나온 중요도 산출 #########################################################################

y = df.DE1_pr # y축 설정
X = df.drop(columns=["DE1_pr"]) #결과축 삭제 for predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 
rnd_clf = RandomForestClassifier(n_estimators=500,n_jobs=-1)
rnd_clf.fit(X_train,y_train)
for name, score in zip(df[:],rnd_clf.feature_importances_): ## RF 적 방법으로 변수 중요도 추출
    print(name, score)

# STEP 7 - step5, step 6의 결론을 토대로 변수 drop ######################################################################

# STEP 8 - 파이프라이닝 - 스케일링 / 모델링 결정  ########################################################################

# SVM , std scaler 로 파이프라이닝 수행 -94
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
pipe.fit(x_train,y_train)
y_preds=pipe.predict(x_test)
pipe.score(x_test,y_test)

# CATBOOST

# GLM

# RF
