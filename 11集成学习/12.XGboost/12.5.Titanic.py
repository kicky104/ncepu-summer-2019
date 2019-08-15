# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import csv
#这里重点在于数据的处理
#age可能有缺失的地方，需要自己来补上

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc_rate = 100 * float(acc.sum()) / a.size
    print ('%s正确率：%.3f%%' % (tip, acc_rate))
    return acc_rate


def load_data(file_name, is_train):
    data = pd.read_csv(file_name)  # 数据文件路径
    pd.set_option('display.width',200)#为了看方便
    # print 'data.describe() = \n', data.describe()#统计所有数值型的实际值

    #分析data.describe():
    #这里survived是0-1编码，加和是所有survived的人数，除以总人数就是存活率；
    #pclass在mean上是2.308642，表示坐3等仓的人数大于1等舱人数；
    # 所以可以把缺失值改为平均值mean这一行的数目；

    # 性别
    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int) #转为int类型的

    # 补齐船票价格缺失值  --  只要发现船票价格是=0的，就把船票等级仓位相同的找到；
    if len(data.Fare[data.Fare == 0]) > 0:
        fare = np.zeros(3)
        for f in range(0, 3):
            fare[f] = data[data.Pclass == f + 1]['Fare'].dropna().median()#把空的先扔掉再取中位数
        for f in range(0, 3):  # loop 0 to 2
            data.loc[(data.Fare == 0) & (data.Pclass == f + 1), 'Fare'] = fare[f] # 'Fare'是用名字索引，所以用loc

    #print ('data.describe()=\n',data.describe())

    # 年龄：使用均值代替缺失值
    # mean_age = data['Age'].dropna().mean()
    # data.loc[(data.Age.isnull()), 'Age'] = mean_age
    if is_train:
        # 年龄：使用随机森林预测年龄缺失值
        print ('随机森林预测缺失年龄：--start--')
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]   # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())] #是我们想预测的部分
        # print age_exist #年龄存在的有714个
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=1000)#这里我们认为年龄是连续的，所以用rf回归的函数
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat #只要为空，就补充为age的预测值
        print ('随机森林预测缺失年龄：--over--')
    else:
        print ('随机森林预测缺失年龄2：--start--')
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]  # 年龄不缺失的数据
        age_null = data_for_age.loc[(data.Age.isnull())]
        # print age_exist
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        # print age_hat
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        print ('随机森林预测缺失年龄2：--over--')

    # 起始城市
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'  # 保留缺失出发城市
    # data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'U': 0}).astype(int) #S和U都是0，被认为是一类
    # print data['Embarked']
    embarked_data = pd.get_dummies(data.Embarked)#获得亚元，看看出发城市都有哪些
        #data.Embarked是SCQU四个值，pd.get_dummies（）帮助做onehot编码？
    print (embarked_data)
    # embarked_data = embarked_data.rename(columns={'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown', 'U': 'UnknownCity'})
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x)) #给标记值取新的名字，对应onehot编码
    data = pd.concat([data, embarked_data], axis=1) #做纵向连接，相当于numpy中某个函数
    print (data.describe())
    data.to_csv('New_Data.csv') #保存成新文件

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]#取新数据
    # x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = None
    if 'Survived' in data:
        y = data['Survived']

    x = np.array(x)
    y = np.array(y)

    # 思考：这样做，其实发生了什么？
    x = np.tile(x, (5, 1))#把x和y个拷贝5份，放在x和y中，会提高分类性能和分裂率
    y = np.tile(y, (5, ))
    if is_train: #训练数据直接返回
        return x, y
    return x, data['PassengerId'] #测试数据当作亚元返回。什么是亚元？？？？？？？？？

#在使用随机森林中给出特征可能不稳定，需要多取特征

def write_result(c, c_type):
    file_name = 'Titanic.test.csv'
    x, passenger_id = load_data(file_name, False)

    if type == 3:
        x = xgb.DMatrix(x)
    y = c.predict(x)
    y[y > 0.5] = 1
    y[~(y > 0.5)] = 0

    predictions_file = open("Prediction_%d.csv" % c_type, "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", "Survived"])
    open_file_object.writerows(zip(passenger_id, y))
    predictions_file.close()


if __name__ == "__main__":
    x, y = load_data('Titanic.train.csv', True) #load_data：是对数据处理好的
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)#测试和验证
    #
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    lr_acc = accuracy_score(y_test, y_hat)
    # write_result(lr, 1)

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_hat = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_test, y_hat)
    # write_result(rfc, 2)

    # XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
             # 'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
    bst = xgb.train(param, data_train, num_boost_round=100, evals=watch_list)#这里 num_boost_round=100取100太高了，一般10就足够了
    y_hat = bst.predict(data_test)
    # write_result(bst, 3)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_acc = accuracy_score(y_test, y_hat)

    print ('Logistic回归：%.3f%%' % lr_acc)
    print ('随机森林：%.3f%%' % rfc_acc)
    print ('XGBoost：%.3f%%' % xgb_acc)
