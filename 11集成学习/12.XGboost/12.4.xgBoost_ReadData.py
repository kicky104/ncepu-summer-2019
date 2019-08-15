# /usr/bin/python
# -*- coding:utf-8 -*-

#简单的小例子

import xgboost as xgb
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def read_data(path):#基本python的读取
    y = []
    row = []
    col = []
    values = []
    r = 0       # 首行
    for d in open(path):
    #for d in file(path):
        d = d.strip().split()      # 以空格分开
        y.append(int(d[0]))
        d = d[1:]
        for c in d:
            key, value = c.split(':')
            row.append(r)
            col.append(int(key))
            values.append(float(value))
        r += 1 #r是第几行，key是第几列
    x = scipy.sparse.csr_matrix((values, (row, col))).toarray()#toarray就转成np格式
    y = np.array(y)
    return x, y


if __name__ == '__main__':
    x, y = read_data('agaricus_train.txt')
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    # Logistic回归
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    print ('Logistic回归正确率：', accuracy_score(y_test, y_hat))

    # XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    print ('XGBoost正确率：', accuracy_score(y_test, y_hat))
