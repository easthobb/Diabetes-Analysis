# STEP 1 -  의존성 로드 ############################################################
import numpy as np
import pandas as pd
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import seaborn as sns # seaborn ref : https://greeksharifa.github.io/machine_learning/2019/12/05/Seaborn-Module/
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_roc_curve
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt # plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

## 정확도 지표 정의
def metrics(y_test,pred):
    print("ACC : ",accuracy_score(y_test,pred))
    print("Precision : ", precision_score(y_test,pred))
    print("recall(TP rate) : ", recall_score(y_test,pred))
    print("F1 : " , f1_score(y_test,pred))
    print("ROC SCORE : ", roc_auc_score(y_test,pred,average="macro"))


#%matplotlib inline
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 200)

## DATA LOAD & Null Inspection
df = pd.read_csv('2018_17_16_OUTCOME.csv')
df.head()

df_0 = df[0:1199]
df_1 = df[1199:2398]
df_2 = df[2398:3597]
df_3 = df[3597:4796]
df_4 = df[4796:5995]
print(len(df_0),len(df_1),len(df_2),len(df_3),len(df_4))

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

fig,ax = plt.subplots()
pipe = Pipeline([('scaler',StandardScaler()), ('classifiers',XGBClassifier())])

### FOLD 0 - TEST / TRAIN SET ###################################
print("\n 0 - FOLD START")
train_df = pd.concat([df_1, df_2, df_3,df_4])
test_df = df_0
X_train = train_df.drop(columns=["HE_DM"])
Y_train = train_df.HE_DM
X_test = test_df.drop(columns=["HE_DM"])
Y_test = test_df.HE_DM

# DO SMOTE ALG
smote = SMOTE(random_state=0)
X_train_over,Y_train_over = smote.fit_sample(X_train,Y_train)
#PIPELINE RESAMPLED WORKFLOW
pipe.fit(X_train_over,Y_train_over)
y_preds=pipe.predict(X_test)
metrics(Y_test,y_preds)
#plot_confusion_matrix(pipe,X_test,Y_test)

#plot set
viz = plot_roc_curve(pipe, X_test, Y_test,
                         name='ROC fold {}'.format(0),
                         alpha=0.3, lw=1, ax=ax)
interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
interp_tpr[0] = 0.0
tprs.append(interp_tpr)
aucs.append(viz.roc_auc)


### FOLD 1 - TEST / TRAIN SET ###################################
print("\n 1 - FOLD START")
train_df = pd.concat([df_0, df_2, df_3,df_4])
test_df = df_1
X_train = train_df.drop(columns=["HE_DM"])
Y_train = train_df.HE_DM
X_test = test_df.drop(columns=["HE_DM"])
Y_test = test_df.HE_DM

# DO SMOTE ALG
smote = SMOTE(random_state=0)
X_train_over,Y_train_over = smote.fit_sample(X_train,Y_train)

#PIPELINE RESAMPLED WORKFLOW
pipe.fit(X_train_over,Y_train_over)
y_preds=pipe.predict(X_test)
metrics(Y_test,y_preds)
#plot_confusion_matrix(pipe,X_test,Y_test)

#plot set
viz = plot_roc_curve(pipe, X_test, Y_test,
                         name='ROC fold {}'.format(1),
                         alpha=0.3, lw=1, ax=ax)
interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
interp_tpr[0] = 0.0
tprs.append(interp_tpr)
aucs.append(viz.roc_auc)


### FOLD 2 - TEST / TRAIN SET ###################################
print("\n 2 - FOLD START")
train_df = pd.concat([df_0, df_1, df_3,df_4])
test_df = df_2
X_train = train_df.drop(columns=["HE_DM"])
Y_train = train_df.HE_DM
X_test = test_df.drop(columns=["HE_DM"])
Y_test = test_df.HE_DM

# DO SMOTE ALG
smote = SMOTE(random_state=0)
X_train_over,Y_train_over = smote.fit_sample(X_train,Y_train)
#PIPELINE RESAMPLED WORKFLOW
pipe.fit(X_train_over,Y_train_over)
y_preds=pipe.predict(X_test)#2
metrics(Y_test,y_preds)
#plot_confusion_matrix(pipe,X_test,Y_test)

#plot set
viz = plot_roc_curve(pipe, X_test, Y_test,
                         name='ROC fold {}'.format(2),
                         alpha=0.3, lw=1, ax=ax)
interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
interp_tpr[0] = 0.0
tprs.append(interp_tpr)
aucs.append(viz.roc_auc)


### FOLD 3 - TEST / TRAIN SET ###################################
print("\n 3 - FOLD START")
train_df = pd.concat([df_0, df_1, df_2,df_4])
test_df = df_3
X_train = train_df.drop(columns=["HE_DM"])
Y_train = train_df.HE_DM
X_test = test_df.drop(columns=["HE_DM"])
Y_test = test_df.HE_DM

# DO SMOTE ALG
smote = SMOTE(random_state=0)
X_train_over,Y_train_over = smote.fit_sample(X_train,Y_train)
#PIPELINE RESAMPLED WORKFLOW
pipe.fit(X_train_over,Y_train_over)
y_preds=pipe.predict(X_test)
metrics(Y_test,y_preds)
print("#3" ,len(y_preds))
#plot_confusion_matrix(pipe,X_test,Y_test)

#plot set
viz = plot_roc_curve(pipe, X_test, Y_test,
                         name='ROC fold {}'.format(3),
                         alpha=0.3, lw=1, ax=ax)
interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
interp_tpr[0] = 0.0
tprs.append(interp_tpr)
aucs.append(viz.roc_auc)


### FOLD 4 - TEST / TRAIN SET ###################################
print("\n 4 - FOLD START")
train_df = pd.concat([df_0, df_1, df_2, df_3])
test_df = df_4
X_train = train_df.drop(columns=["HE_DM"])
Y_train = train_df.HE_DM
X_test = test_df.drop(columns=["HE_DM"])
Y_test = test_df.HE_DM

# DO SMOTE ALG
smote = SMOTE(random_state=0)
X_train_over,Y_train_over = smote.fit_sample(X_train,Y_train)
#PIPELINE RESAMPLED WORKFLOW
pipe.fit(X_train_over,Y_train_over)
y_preds=pipe.predict(X_test)
metrics(Y_test,y_preds)
print("#4" ,len(y_preds))
#plot_confusion_matrix(pipe,X_test,Y_test)

#plot set
viz = plot_roc_curve(pipe, X_test, Y_test,
                         name='ROC fold {}'.format(3),
                         alpha=0.3, lw=1, ax=ax)
interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
interp_tpr[0] = 0.0
tprs.append(interp_tpr)
aucs.append(viz.roc_auc)


#############################
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="XGBOOST ROC CURVE 5-fold")
ax.legend(loc="lower right")

plt.show()
