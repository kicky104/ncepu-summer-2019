#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#from sklearn.pipeline import accuracy_score
import pydotplus #前身是pydot，专门生成dot文件，也只用于这里

#使用精确度做判断

# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = 'iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x = data[range(4)] #0，1，2，3列给x
    y = pd.Categorical(data[4]).codes #最后一列是字符串型的，先转为编码的形式
    #print y
    # 为了可视化，仅使用前两列特征
    x = x.iloc[:, :2]#也可以写为：x=[x[0,1]],表示只读取前两列数据；第一个:表示所有行都要，:2表示从0列到第二列，且不包含第二列；
        # iloc表示按索引读取（也可写作ix），而不是按名字读取；
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)#划分训练数据和测试数据
    print y_test.shape

    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    model = DecisionTreeClassifier(criterion='entropy') #函数默认参数：max_depth 树的最大深度，并不是越深越好；
        #默认参数：min_samples_split：最小可以分割的节点的样本个数；min_samples_leaf：叶节点所包含的最小样本个数；
        #决策树分类器，是在sklearn.tree 这个包下
        #这里默认分类器的分类标准criterion是entropy；也可改作criterion='GINI'；
    model.fit(x_train, y_train) #训练数据给model，fit一下；
    #这里，并不是调包不需要它，我们就不学原理，
    y_test_hat = model.predict(x_test)      # 测试数据，得到估计值
#    print 'accuracy_score:',accuracy_score(y_test,y_test_hat) #看当前测试值

    # 保存
    # dot -Tpng my.dot -o my.png
    # 1、输出
    with open('iris.dot', 'w') as f: #输出成一个dot格式的文件，f是文件指针
        tree.export_graphviz(model, out_file=f )#强制
    # 2、给定文件名
    # tree.export_graphviz(model, out_file='iris1.dot')
    # 3、输出为pdf格式
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E, class_names=iris_class,
                                    filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris.pdf') #graph可以直接输出成pdf，也可以输出成图片类型
    f = open('iris.png', 'wb')
    f.write(graph.create_png())
    f.close()

    #one hot编码在决策树时不是很常见，6.6LR代码使用到one hot代码

    # 画图
    N, M = 50, 50  # 横纵各采样多少个值
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
    print x_show.shape

    # # 无意义，只是为了凑另外两个维度
    # # 打开该注释前，确保注释掉x = x[:, :2]
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])#样本给3个颜色
    y_show_hat = model.predict(x_show)  # 预测值
    print y_show_hat.shape
    print y_show_hat
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    print y_show_hat
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示，画出色块
    plt.scatter(x_test[0], x_test[1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=cm_dark, marker='*')  # 测试数据
    plt.scatter(x[0], x[1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据，marker默认使用圆圈
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
    plt.show()

    # 训练集上的预测结果
    y_test = y_test.reshape(-1) #测试数据拉成直线
    print y_test_hat
    print y_test
    result = (y_test_hat == y_test)   #将测试数据和真实数据进行比对 ，True则预测正确，False则预测错误
    acc = np.mean(result)#true和false的均值就是精确度
    print '准确度: %.2f%%' % (100 * acc) #这里写了一个准确度测试

    # 过拟合：错误率
    depth = np.arange(1, 15) #给从1-14每个值做测试
    err_list = []#错误率列表
    for d in depth: #将树的深度与错误率结合作图，找出二者关系，当深度=3时，错误率最高 ---- 交叉验证
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        if d == 1:
            print result
        err = 1 - np.mean(result)
        err_list.append(err)
        # print d, ' 准确度: %.2f%%' % (100 * err)
        print d, ' 错误率: %.2f%%' % (100 * err)
    plt.figure(facecolor='w')
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()
