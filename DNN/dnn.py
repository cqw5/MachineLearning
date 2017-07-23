# -*- coding: utf-8 -*-
# Author: qwchen
# Date: 2017-06-22

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras import metrics
from keras.models import load_model
import h5py
import gc

def save(file, instanceID, pred):
    """
        保存预测结果
    """
    with open(file ,'w') as f:
        f.write('instanceID,pred\n')
        for i in range(len(instanceID)):
            line = '{0},{1}\n'.format(instanceID[i], pred[i])
            f.write(line)

def logloss(act, pred):
    """
        评估结果
    """
    act = np.array(act)
    pred = np.array(pred)
    epsilon = 1e-15
    pred = np.maximum(epsilon, pred)
    pred = np.minimum(1.0 - epsilon, pred)
    ll = sum(act * np.log(pred) + (1.0 - act) * np.log(1.0 - pred))
    ll = ll * -1.0 / len(act)
    return ll

def generate_mini_batch_train_data(file, miniBatch=64):
    """
        读取训练数据的生成器
    """
    i = 0
    feas = []
    labels = []
    while 1:
        f = open(file)
        f.readline()
        for line in f:
            content = line.strip().split(',')
            fea = map(float, content[2:])
            label = int(content[1])
            if i < miniBatch:
                feas.append(fea)
                labels.append(label)
                i += 1
            else:
                yield np.array(feas), np.array(labels)
                feas = []
                labels = []
                i = 0
        f.close()

def generate_all_valid_data(file):
    """
        读取验证数据的生成器
    """
    df_validData = pd.read_csv(file)
    feas = df_validData[columns].values
    labels = df_validData['label'].values
    while 1:
        yield feas, labels

def get_fea(file):
    """
        获取特征
    """
    columns = []
    with open(file, 'r') as f:
        print 'Used Feature:'
        for line in f.readlines()[2:]:
            line = line.strip()
            if line[0] != '#':
                print line
                columns.append(line)
    print 'Feature Num: {0}'.format(len(columns))
    return columns

def get_model(feas_num):
    """
        获取模型
    """
    # DNN模型结构
    model = Sequential()
    # 第1个隐含层
    model.add(Dense(512, input_dim=feas_num, activation='relu'))
    # model.add(Dropout(0.5))
    # 第2个隐含层
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # 第3个隐含层
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # 第4个隐含层
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # 第5个隐含层
    model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.5))
    # 输出层
    model.add(Dense(1, activation='sigmoid'))
    # 参数
    model.compile(loss='binary_crossentropy',
              optimizer='SGD',   # SGD, Adam
              metrics=[metrics.binary_crossentropy])
    return model

def train(fileTrain, fileTest, model, miniBatch, steps_per_epoch, epochs):
    """
        训练模型
    """
    model.fit_generator(generate_mini_batch_train_data(fileTrain, miniBatch=miniBatch),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generate_all_valid_data(fileTest),
        validation_steps=1,
        verbose=1,
        max_q_size=100,
        workers=1)
    return model

def predict(filePred, fileRes, model, columns):
    """
        生成结果
    """
    df = pd.read_csv(filePred)
    instanceID = df['instanceID'].values
    feas = df[columns].values
    pred = model.predict_proba(feas)
    # print 'Pred result: 10'
    # print pred[:10]
    save(fileRes, instanceID, pred[:,0])

def train_all(fileTrain, fileTest, filePred, fileRes, fileModel, columns, miniBatch, steps_per_epoch, epochs):
    """
        模型一次训练到底
    """
    model = get_model(len(columns))
    model = train(fileTrain, fileTest, model, miniBatch, steps_per_epoch, epochs)
    predict(filePred, fileRes, model, columns)


def train_incre(fileTrain, fileTest, filePred, fileRes, fileModel, columns, miniBatch, steps_per_epoch, epochs):
    """
        模型增量训练
    """
    for i in range(epochs):
        print "Epoch {0}".format(i)
        if i == 0:
            model = get_model(len(columns))
            model = train(fileTrain, fileTest, model, miniBatch, steps_per_epoch, 1)
            curModel = '{0}_{1}'.format(fileModel, i)
            curRes = '{0}_{1}'.format(fileRes, i)
            print 'Save model {0}'.format(curModel)
            model.save(curModel)
            print 'Pred {0}'.format(curRes)
            predict(filePred, curRes, model, columns)
            del model
            gc.collect()
        else:
            # 每一轮训练前，先对训练数据做shuffle
            # train_data = pd.read_csv(fileTrain)
            # train_data = shuffle(train_data)
            # train_data.to_csv(fileTrain, index=False, header=True)
            # del train_data
            # gc.collect()
            # 加载前一轮的模型
            preModel = '{0}_{1}'.format(fileModel, i-1)
            model = load_model(preModel)
            model = train(fileTrain, fileTest, model, miniBatch, steps_per_epoch, 1)
            curModel = '{0}_{1}'.format(fileModel, i)
            curRes = '{0}_{1}'.format(fileRes, i)
            model.save(curModel)
            predict(filePred, curRes, model, columns)
            del model
            gc.collect()

if __name__ == '__main__':
    train_data_num = 17147838
    miniBatch = 64
    steps_per_epoch = train_data_num / miniBatch
    # steps_per_epoch = 100
    epochs = 10
    srcPath = '../preData/dnn_bin_all/'
    dstPath = '../preData/dnn_bin_all/miniBatch_{0}_epochs_{1}_incre/'.format(miniBatch, epochs)
    fileDict = srcPath + 'fea_dict.csv'
    fileTrain = srcPath + 'train.csv'
    fileTest = srcPath + 'valid.csv'
    filePred = srcPath + 'pred.csv'
    fileModel = dstPath + 'model'
    fileRes = dstPath + 'pred'
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    columns = get_fea(fileDict)
    # train_all(fileTrain, fileTest, filePred, fileRes, fileModel, columns, miniBatch, steps_per_epoch, epochs)
    train_incre(fileTrain, fileTest, filePred, fileRes, fileModel, columns, miniBatch, steps_per_epoch, epochs)


