# coding: utf-8

import os  # 用于文件名和路径处理
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# from sklearn.preprocessing import Imputer  # sklearn < 0.20
from sklearn.impute import SimpleImputer  # sklearn >= 0.20

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, roc_curve, \
    confusion_matrix, make_scorer
from sklearn.metrics import mean_squared_error, zero_one_loss, log_loss

COL_INPUT = [
    'haloperidol', 'sex', 'age', 'smoking', 'history PONV', 'combined Anesthesia', 'anesthesia Duration',
    'sufentanil Bolus', 'sufentanil Epidural', 'sufentanil Infusion', 'sufentanil TCI', 'remifentanil Bolus',
    'remifentanil Infusion', 'remifentanil TCI', 'piritramide'
]
# COL_STR = []  # ['risk']
COL_Y = None

enc = OneHotEncoder()
ss = MinMaxScaler()


def read_csv(fp, encoding='gb18030', y_num=1, col_input=None, col_str=None):
    pocd = pd.read_csv(fp, encoding=encoding)
    pocd = do_base_preprocessing(pocd, y_num=y_num, col_input=col_input, col_str=col_str)  # 覆盖掉原来的pocd

    return pocd


def do_base_preprocessing(pocd, with_y=True, y_num=1, col_input=None, col_str=None):
    global COL_INPUT, COL_Y
    if col_input is None:
        col_input = COL_INPUT
    if col_str is None:
        col_str = []
    COL_Y = [str(i) for i in pocd.columns[-y_num:]]
    # print("col_input", col_input, "COL_Y", COL_Y)  # TODO

    new_pocd = pocd
    if len(col_str) > 0:
        if with_y:
            enc.fit(pocd[col_str])
        t = enc.transform(pocd[col_str]).toarray()
        new_pocd = pocd.drop(col_str, axis=1).join(pd.DataFrame(t, columns=enc.get_feature_names(col_str)))

    y = None
    if with_y:
        y = pocd[COL_Y]
        new_pocd = new_pocd.drop(COL_Y, axis=1)
    columns = new_pocd.columns
    # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)  # sklearn < 0.20：配置缺失数据处理模型
    imp = SimpleImputer(missing_values=np.NaN, strategy='mean')  # sklearn >= 0.20：改用SimpleImputer
    new_pocd = imp.fit_transform(new_pocd)  # 对数据进行“训练”（处理），得到转换后的数据
    new_pocd = pd.DataFrame(new_pocd)  # 转换为pd的DataFrame
    new_pocd.columns = columns  # 恢复表头
    if with_y:
        new_pocd_list = []
        for sy in COL_Y:
            p = new_pocd.join(y[[sy]])
            new_pocd_list.append(p)
        return new_pocd_list if len(new_pocd_list) > 1 else new_pocd_list[0]

    return new_pocd


def do_split_data(pocd, random_state=1):
    # 原始数据的处理（分割X, y，抽样分割，等等）
    X, y = split_x_y(pocd)

    # 仅分成2份：训练集，测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)

    # reshape操作
    X_train, y_train = np_reshape_x_y(X_train, y_train)
    # X_validate, y_validate = np_reshape_x_y(X_validate, y_validate)
    X_test, y_test = np_reshape_x_y(X_test, y_test)

    return X_train, X_test, y_train, y_test


def do_xy_preprocessing(X_train, X_test, y_train, y_test):
    # minmaxscaler归一化（对X做就可以，用train的fit，对train、validate、test进行transform）
    X_train = ss.fit_transform(X_train)
    # X_validate = ss.transform(X_validate)
    X_test = ss.transform(X_test)

    # print(X_train)
    # print(X_validate)
    # print(X_test)

    return X_train, X_test, y_train, y_test


def do_predict_preprocessing(pocd_predict):
    return ss.transform(pocd_predict)


# 通用的X, y列分割
def split_x_y(df):
    # 分割出 (变量列, 目标列)
    return df.iloc[:, 0:-1], df.iloc[:, -1:]


# 通用的训练/预测前的数据reshape操作
def np_reshape_x_y(X, y):
    X = np.array(X)
    y = np.array(y)
    ya, yb = y.shape
    y = y.reshape(ya, )
    return X, y


# 通用的模型评分
def model_score(model, X, y):
    model_result = {}
    y_proba = model_result["y_proba"] = model.predict_proba(X)
    y_pre = model_result["y_pre"] = model.predict(X)
    score = model_result["score"] = model.score(X, y)
    acc_score = model_result["accuracy_score"] = accuracy_score(y, y_pre)
    preci_score = model_result["preci_score"] = precision_score(y, y_pre)
    rec_score = model_result["recall_score"] = recall_score(y, y_pre)
    f1__score = model_result["f1_score"] = f1_score(y, y_pre)
    auc = model_result["auc"] = roc_auc_score(y, y_proba[:, 1])

    mse = model_result["mse"] = mean_squared_error(y, y_pre)
    zero_one_loss_fraction = model_result["zero_one_loss_fraction"] = zero_one_loss(y, y_pre, normalize=True)
    zero_one_loss_num = model_result["zero_one_loss_num"] = zero_one_loss(y, y_pre, normalize=False)

    con_matrix = model_result["confusion_matrix"] = confusion_matrix(y, y_pre, labels=[0, 1])

    fpr, tpr, threasholds = model_result["fpr"], model_result["tpr"], model_result["threasholds"] = roc_curve(y,
                                                                                                              y_proba[:,
                                                                                                              1])

    # 增加交叉验证分数！（取n折交叉后的平均分数）
    scorer = make_scorer(roc_auc_score)
    scores = cross_val_score(model, X, y, scoring=scorer, cv=5, n_jobs=1)
    cv_score = model_result["cv_score"] = np.mean(scores)

    return model_result


# 通用的模型训练+评分
def model_fit_score(model, X, y):
    model.fit(X, y)

    model_result = model_score(model, X, y)
    return model_result


# 通用的模型结果打印
def model_print(model_result, model_name):
    acc_score = model_result["accuracy_score"]
    preci_score = model_result["preci_score"]
    rec_score = model_result["recall_score"]
    f1__score = model_result["f1_score"]
    auc = model_result["auc"]

    mse = model_result["mse"]
    zero_one_loss_fraction = model_result["zero_one_loss_fraction"]
    zero_one_loss_num = model_result["zero_one_loss_num"]

    cv_score = model_result["cv_score"]

    con_matrix = model_result["confusion_matrix"]

    return '[%s] accuracy_score: %.3f, preci_score: %.3f, recall_score: %.3f, f1_score: %.3f, auc: %.3f,\n mse: %.3f, zero_one_loss_fraction: %.3f, zero_one_loss_num: %.3f, cv_score: %.3f\nconfusion_matrix:' % (
        model_name, acc_score, preci_score, rec_score, f1__score, auc, mse, zero_one_loss_fraction,
        zero_one_loss_num,
        cv_score) + str(con_matrix) + '\n'


# 通用画roc/auc图
def plt_roc_auc(model_result_list, title):
    # color_list = ['r', 'b', 'darkblue', 'g', 'y', 'purple']
    fig = plt.figure(figsize=(6, 6), dpi=150)
    for i, model_item in enumerate(model_result_list):
        model_result, model_name = model_item
        fpr, tpr, threasholds = model_result["fpr"], model_result["tpr"], model_result["threasholds"]
        auc = model_result["auc"]
        plt.plot(fpr, tpr, lw=2, label='%s, AUC=%.3f' % (model_name, auc))  # , c=color_list[i%len(color_list)]

    plt.plot((0, 1), (0, 1), c='#a0a0a0', lw=2, ls='--')
    plt.xlim(-0.001, 1.001)
    plt.ylim(-0.001, 1.001)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(title, fontsize=18)
    return fig
