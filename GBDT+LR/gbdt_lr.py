# -*- coding: utf-8 -*-
# Author: qwchen
# Date: 2017-07-22
#
# 通过树模型来构造组合特征的方法
# - RandomForest + LR
# - GBDT + LR: FaceBook Paper
# 为了防止过拟合，训练Tree Model的样本和用Tree Model生成特征的样本(即训练LR的样本)应该是不同的样本

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


def rf_lr_model():
    """
    RandomForest + LR
    """
    rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    y_pred = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    print 'RandomForest+LR AUC: {0}'.format(auc(fpr, tpr))


def gbdt_lr_model():
    """
    GBDT + LR
    """
    gbdt = GradientBoostingClassifier(n_estimators=n_estimator, max_depth=max_depth)
    gbdt_enc = OneHotEncoder()
    gbdt_lm = LogisticRegression()
    gbdt.fit(X_train, y_train)
    gbdt_enc.fit(gbdt.apply(X_train)[:, :, 0])
    gbdt_lm.fit(gbdt_enc.transform(gbdt.apply(X_train_lr)[:, :, 0]), y_train_lr)
    y_pred = gbdt_lm.predict_proba(gbdt_enc.transform(gbdt.apply(X_test)[:, :, 0]))[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    print 'GBDT+LR AUC: {0}'.format(auc(fpr, tpr))
    # 可以输出样本的特征经过GBDT转换后的结果，输出的是稀疏编码，可以通过toarray()转换为稠密编码
    # gbdt_enc.transform(gbdt.apply(X_train_lr)[:, :, 0])            # 稀疏编码
    # gbdt_enc.transform(gbdt.apply(X_train_lr)[:, :, 0]).toarray()  # 稠密编码


def lr_model():
    """
    LR
    """
    lr = LogisticRegression()
    lr.fit(X_train_lr, y_train_lr)
    y_pred = lr.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    print 'LR AUC: {0}'.format(auc(fpr, tpr))


def rf_model():
    """
    RandomForest
    """
    rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
    rf.fit(X_train_lr, y_train_lr)
    y_pred = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    print 'RandomForest AUC: {0}'.format(auc(fpr, tpr))


def gbdt_model():
    """
    GBDT
    """
    rf = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth)
    rf.fit(X_train_lr, y_train_lr)
    y_pred = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    print 'GBDT AUC: {0}'.format(auc(fpr, tpr))


n_estimator = 30
max_depth = 6
X, y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train, test_size=0.5)
lr_model()
rf_model()
gbdt_model()
rf_lr_model()
gbdt_lr_model()

