# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#造数据
def f(x):
    return 0.5*np.exp(-(x+3) **2) + np.exp(-x**2) + 0.5*np.exp(-(x-3) ** 2)


if __name__ == "__main__":
    np.random.seed(0)
    N = 200
    x = np.random.rand(N) * 10 - 5  # [-5,5)
    x = np.sort(x)
    y = f(x) + 0.05*np.random.randn(N)
    x.shape = -1, 1

    degree = 6 #用PolynomialFeatures多少个特征
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 20), fit_intercept=False) #RidgeCV：可以调参的岭回归，alphas是参数值，这里取值很广；得到回归本身的模型
        # fit_intercept=False，因为这里1认为已经有了
    ridged = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('Ridge', ridge)]) #这里先对数据做6阶的Polynomial，在做岭回归，二者构成一个pipeline
    bagging_ridged = BaggingRegressor(ridged, n_estimators=100, max_samples=0.2)#ridged作为基本分类器，给bagging，用BaggingRegressor做100个分类器，
    dtr = DecisionTreeRegressor(max_depth=5)#正常决策树
    regs = [
        ('DecisionTree Regressor', dtr),#普通决策树
        ('Ridge Regressor(%d Degree)' % degree, ridged), #几度的岭回归
        ('Bagging Ridge(%d Degree)' % degree, bagging_ridged), #有bagging后的岭回归
        ('Bagging DecisionTree Regressor', BaggingRegressor(dtr, n_estimators=100, max_samples=0.2))] #100棵决策树生成的BaggingRegressor，0.2的采样率
    x_test = np.linspace(1.1*x.min(), 1.1*x.max(), 1000)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.plot(x, y, 'ro', label=u'训练数据')
    plt.plot(x_test, f(x_test), color='k', lw=3.5, label=u'真实值')
    clrs = 'bmyg'
    for i, (name, reg) in enumerate(regs):
        reg.fit(x, y)
        y_test = reg.predict(x_test.reshape(-1, 1))
        plt.plot(x_test, y_test.ravel(), color=clrs[i], lw=i+1, label=name, zorder=6-i)
    plt.legend(loc='upper left')
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.title(u'回归曲线拟合', fontsize=21)
    plt.ylim((-0.2, 1.2))
    plt.tight_layout(2)
    plt.grid(True)
    plt.show()
