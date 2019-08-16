# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw

#数据是关于蘑菇的，第一列表示是否有毒，1-有毒；每行中，写为1的属性是1，其他属性值是0，实质是onehot编码；
#每行数据可表示为：00100000110000000001...
#agarious是关于数据的说明，共有125个特征
#非数值型编码有时可以改为one hot编码

# 定义f: theta * x
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label() #这里是负梯度
    h = p * (1.0-p) #p是个sigmod函数，p对theta求导，是这个函数值
    return g, h#自己定义g和h，这里其实就是logistic回归的损失函数
    #g和h是一种计算方式，因为函数已知；


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)
    #y.get_label()就可以区分x和y的值；


if __name__ == "__main__":

    # 读取数据
    data_train = xgb.DMatrix('agaricus_train.txt')#读到xgboost中
    data_test = xgb.DMatrix('agaricus_test.txt')
    print (data_train)
    print (type(data_train))
    #这里就是1个0-1的二分类问题

    # 设置参数
    param = {'max_depth': 3, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'} # logitraw，eta=1表示有若干个基本分类器，需要给定每一棵树的权值，，eta没有防止过拟合的问题，正常的xgboost
        #学习率给0.2、0.3即可；max_depth：每棵决策树都是3层的；
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]#每次得到基本分类器上都在训练数据和测试数据上看精度有多少；为了知道在什么时候停止增加分类器
    n_round = 7 #xgboost和adaboost，决策树不需要那么多，不需要像随机森林一样，需要100棵数
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)#param是前面设置好的参数，data_train包含x和y两列数据；
        #num_boost_round：一共做几轮，也就是有几棵树；evals：是得到的数据，用来和真实y值作比较；obj：做的某个对象，可以没有；feval=error_rate是自己给的错误率的计算方式；
        #有时也可能自定义1阶导数和2阶导数；
    #min_child_weight:是指当子节点的权值小于某一个值是就不要再分割了；如果希望模型有更好的泛化能力，则可以将这个值调得稍微大一点；

    # 计算错误率
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print (y_hat)
    print (y)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print ('样本总数：\t', len(y_hat))
    print ('错误数目：\t%4d' % error)
    print ('错误率：\t%.5f%%' % (100*error_rate))
