# /usr/bin/python
# -*- encoding:utf-8 -*-
#3分类-softmax
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

'''
def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]
'''
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

if __name__ == "__main__":
    path = 'iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    data = pd.read_csv(path, header=None)
    x,y = data[range(4)],data[4]
    y = pd.Categorical(y).codes #取出编码
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=50)#随机分为测试数据和训练数据

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 2, 'eta': 0.3, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}# 'silent': 1,输出每一个值是不是要剪枝等信息

    bst = xgb.train(param, data_train, num_boost_round=6, evals=watch_list)#使用前面参数做训练，做6轮
    y_hat = bst.predict(data_test)#用test数据做预测
    print (y_hat)
    print ('y_hat==y_test=',y_hat==y_test)
    result = y_test == y_hat#后面两项先做比较，相等与不等再赋值给realut
    print ('正确率:\t', float(np.sum(result)) / len(y_hat))
    print ('END.....\n')

    #logistic回归中有个penoty，是L1和L2的正则
    models=[('LogisticRegression',LogisticRegressionCV(Cs=10,cv=3)),
            ('RandomForest',RandomForestClassifier(n_estimators=30,criterion='gini'))]
    for name,model in models:
        model.fit(x_train,y_train)
        print (name,'训练集正确率',accuracy_score(y_train,model.predict(x_train)))
        print (name,'测试集正确率',accuracy_score(y_test,model.predict(x_test)))
