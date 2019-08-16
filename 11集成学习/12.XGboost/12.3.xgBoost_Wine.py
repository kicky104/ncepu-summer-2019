# !/usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split   # cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#第1列表示类别；后面数据是葡萄酒的各个属性；文件是以，隔开的属性
if __name__ == "__main__":
    # 作业：尝试用Pandas读取试试？
    data = np.loadtxt('wine.data', dtype=float, delimiter=',')#用np读取数据，delimiter是分隔符
    y, x = np.split(data, (1,), axis=1) # (1,)：从0开始到1且不包含1分为1份；从1-最后分成一份；因此这里是两份数据的划分；axis=1：轴是1表示要竖着分类
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.5)

    # Logistic回归
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    print ('Logistic回归正确率：', accuracy_score(y_test, y_hat))

    # XGBoost
    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    #这里是因为softmax要求标记必须从0开始，但是数据读出来标记是从1开始，因此这里要强制将标记为3的改为0；这是属性标号是1，2，0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    params = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(params, data_train, num_boost_round=2, evals=watch_list)
    y_hat = bst.predict(data_test)
    print ('XGBoost正确率：', accuracy_score(y_test, y_hat))
