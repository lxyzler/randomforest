import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,LabelBinarizer,LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold,train_test_split,cross_val_score



log_data=pd.read_csv('/home/joker/MSBD5002_assignment2/MSBD5002_assignment2/data/trainFeatures.csv',encoding='utf-8',na_values='NAN')
trainLabels=pd.read_csv('/home/joker/MSBD5002_assignment2/MSBD5002_assignment2/data/trainLabels.csv',encoding='utf-8',header=None)
test=pd.read_csv('/home/joker/MSBD5002_assignment2/MSBD5002_assignment2/data/testFeatures.csv',encoding='utf-8')
t=pd.read_csv('/home/joker/MSBD5002_assignment2/MSBD5002_assignment2/data/sampleSubmission.csv',encoding='utf-8')
log_data['workclass']=log_data['workclass'].map(str.strip)
log_data['occupation']=log_data['occupation'].map(str.strip)
log_data['native-country']=log_data['native-country'].map(str.strip)
log_data=log_data.replace(' ','')

log_data=log_data.replace('?',np.NaN)
m=log_data['workclass'].mode()
log_data['workclass']=log_data['workclass'].fillna(log_data['workclass'].mode()[0])
log_data['occupation']=log_data['occupation'].fillna(log_data['occupation'].mode()[0])
log_data['native-country']=log_data['native-country'].fillna(log_data['native-country'].mode()[0])

for col in log_data.columns:
    if log_data[col].dtype==object:
        log_data[col]=LabelEncoder().fit_transform(log_data[col])
#log_data
x=log_data
y=trainLabels.values
y=y.ravel()
t=t.values
t=t.ravel()

test=test.replace(' ?',np.NaN)
test.isnull()
test=test.replace('?',np.NaN)
test=test.fillna(np.NaN)
test['workclass']=test['workclass'].fillna(test['workclass'].mode()[0])
test['occupation']=test['occupation'].fillna(test['occupation'].mode()[0])
test['native-country']=test['native-country'].fillna(test['native-country'].mode()[0])

for col in test.columns:
    if test[col].dtype==object:
        test[col]=LabelEncoder().fit_transform(test[col])


#选择特征
feat_labels=log_data.columns
clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
clf = clf.fit(x, y)
importances=clf.feature_importances_
indices=np.argsort(importances)[::-1]
for f in range(x.shape[1]): 
    #给予10000颗决策树平均不纯度衰减的计算来评估特征重要性 
    #print ("%2d) %-*s %f" % (f+1,30,feat_labels[f],importances[indices[f]]) ) 
    #可视化特征重要性-依据平均不纯度衰减
    #plt.bar(range(x.shape[1]),importances[indices],color='lightblue',align='center') 
    #plt.xticks(range(x.shape[1]),feat_labels,rotation=90) 
    #plt.xlim([-1,x.shape[1]]) 
    #plt.tight_layout() 
    #plt.show()
    #print(feat_labels[f],importances[indices[f]])
    if importances[indices[f]]<=0:
        pass
        del log_data[feat_labels[f]]
        del test[feat_labels[f]]
x=log_data
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

bdt=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
bdt.fit(X_train, y_train)

testlabel=bdt.predict(test)

testlabel=pd.DataFrame(testlabel)

testlabel.to_csv('A2_xmaoad_20548149_prediction.csv',columns=None,header=None,index=None)

print(bdt.score(X_test,y_test))
