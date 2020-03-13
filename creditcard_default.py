# -*- coding: utf-8 -*-
#对信用卡违约的预测判断
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


#观察违约的数量，可视化
data = pd.read_csv('.\\UCI_Credit_Card.csv')
print(data.shape)
print(data.describe())
print(data.axes)
n_month = data['default.payment.next.month'].value_counts()
print(n_month)
df = pd.DataFrame({'default.payment.next.month':n_month.index,'values':n_month.values})
plt.figure(figsize=(6,6),dpi=80)
plt.rcParams['font.sans-serif'] = ['simHei']
sns.set_color_codes('pastel')
sns.barplot(x = 'default.payment.next.month',y = 'values',data=df)
plt.title('信用卡违约客户\n(违约：1，守约：0)')
plt.ylabel('人数')
locs , labels = plt.xticks()
plt.show()

#数据处理，特征选择  提取feature,target
data.drop('ID',inplace=True,axis=1)
target = data['default.payment.next.month'].values
columns = data.columns.tolist()
columns.remove('default.payment.next.month')
features = data[columns].values
train_x , test_x, train_y,test_y = train_test_split(features,target,test_size=0.2)

# 定义各种分类器算法
classifiers = [
    SVC(random_state=1,kernel='rbf'),
    DecisionTreeClassifier(random_state=1,criterion='gini'),
    RandomForestClassifier(random_state=1,criterion='gini'),
    KNeighborsClassifier(algorithm='auto'),
    AdaBoostClassifier()
]

#分类器名称
classifier_names = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'kneighborsclassifier',
    'adaboostclassifier'
]
#分类器参数
classifier_param_gird = [
    {"svc__C":[1],'svc__gamma':[0.01]},
    {'decisiontreeclassifier__max_depth':[6,9,11]},
    {'randomforestclassifier__n_estimators':[3,5,7]},
    {'kneighborsclassifier__n_neighbors':[4,6,8]},
    {'adaboostclassifier__n_estimators':list(range(10,100,20))}
]
#使用pipeline

parameters = {"adboostclassifize__n_estimators":[10,50,100]}
# 使用GridsearchCV
def GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,param_grid,score = 'accuracy'):
    response = {}
    # 寻找最优的参数 和最优的准确率分数
    gridsearch = GridSearchCV(estimator=pipeline ,param_grid=param_grid,scoring=score)
    clf = gridsearch.fit(train_x,train_y)
    print('GridSearch最优参数:',clf.best_params_)
    print('GridSearch最优分数:%0.4lf'%clf.best_score_)
    predict_y = clf.predict(test_x)
    print('准确率 :%0.4lf'%accuracy_score(test_y,predict_y))
    response["predict_y"] = predict_y
    response['accuracy_score'] = accuracy_score(test_y,predict_y)
    return response


for model , model_name , model_param_grid in zip(classifiers,classifier_names,classifier_param_gird):
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        (model_name,model)
    ])
    result = GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,model_param_grid,score = 'accuracy')

# @Time : 2020/2/20 16:17
# @Author : pingguo
# @File : creditcard_default.py.py
