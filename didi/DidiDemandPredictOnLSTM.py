# -*- coding: utf-8 -*-

"""
@Time    2023-02-03：
@Author  ：
@File    ：
@Version ：1.0
@Function：
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
需求说明：(7)构建回归预测模型、遗传算法优化的神经网络预测模型和基于两种单项模型的组合预测模型,
           通过对载客热点区内出租车出行需求预测分析,选取模型评价指标对模型的预测效果进行评估。
程序说明：
1). 使用LSTM神经网络对工作日热点区域全时段出现需求进行预测
2). 程序包括：1. 数据加载和预处理
            2. LSTM模型构建
            3. LSTM模型训练和保存
            4. LSTM模型评估：使用r2分数（介于0-1之间，越大回归效果越好），平均绝对误差mae，均方误差mse进行评估
            5. LSTM模型重新加载和新样本预测
            6. 还包括对模型真实值和预测值的可视化。
'''
# 1. 数据加载和数据预处理函数
"""
@:param  :  infile_path 文件路径
@:param  ：  sequence_length LSTM序列长度
@:param  ：  split 训练和测试数据划分
"""
def load_data(infile_path, sequence_length=10, split=0.9):
    # 读取打车需求量数据
    #infile_path = "./data/demand"
    # 只保留工作日数据
    work_days = ['demand_2016.08.08_110100_.csv', 'demand_2016.08.09_110100_.csv',
                 'demand_2016.08.10_110100_.csv', 'demand_2016.08.11_110100_.csv',
                 'demand_2016.08.12_110100_.csv']

    indx = 1
    data = pd.DataFrame()
    for file in work_days:
        infile = os.path.join(infile_path, file)
        print('infile= ', infile)
        df = pd.read_csv(infile, usecols=['hour', 'longitude', 'latitude', 'value'])
        df['day'] = indx  #添加星期列
        indx += 1
        data = pd.concat([data, df], axis=0)

    # 重新构建数据索引
    data.reset_index(drop=True, inplace=True)

    data['value'].describe()

    data_new = data.groupby(['day', 'hour'], as_index=False).agg({'value': 'sum'})
    # 取周三数据进行分析
    #data_1 = data_new[data_new['day'] == 3][['hour', 'value']].to_numpy()

    print('data_1: ', data_new)

    # 对特征列进行归一化，防止绝对量纲对模型影响，同时加快训练速度
    sc = MinMaxScaler()
    data_all = sc.fit_transform(data_new)
    # 构建用于LSTM模型的输入样本
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        # 当前元素位置加序列长度，正好组成样本
        data.append(data_all[i: i + sequence_length + 1])  # sequence_length  是序列长度
    # print(data)
    reshaped_data = np.array(data).astype('float64')
    np.random.shuffle(reshaped_data)
    x = reshaped_data[:, :, :-1]
    y = reshaped_data[:, -1, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[:split_boundary]
    test_x = x[split_boundary:]
    train_y = y[:split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y, sc

# 2.定义LSTM组合模型的函数
def build_model():
    # 使用Sequtial()方式构建LSTM模型
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=100, activation='tanh', return_sequences=True))
    print(lstm_model.layers)
    lstm_model.add(LSTM(units=100, return_sequences=False))
    lstm_model.add(Dense(units=1))
    lstm_model.add(Activation(activation='linear'))
    # 使用损失函数，评估指标，优化器编译模型
    lstm_model.compile(loss='mse', metrics='mse', optimizer='adam')
    return lstm_model  #返回编译好的模型

# 2.训练LSTM模型函数
"""
@:param  :   train_x   训练数据
@:param  ：  train_y   训练数据对应值
@:param  ：  test_x    训练数据
@:param  ：  test_y    测试数据对应值
"""
def train_model(train_x, train_y, test_x, test_y):
    lstm_model = build_model()
    # 设置批数据大小，使用整个数据集多少次epochs，训练过程中验证集比例
    lstm_model.fit(train_x, train_y, batch_size=10, epochs=100, validation_split=0.1)
    saved_path = './lstm_model'
    # 保存训练好的模型
    lstm_model.save(filepath=saved_path, overwrite=True, save_format='tf')
    return lstm_model, saved_path  #返回编译好的模型

# 3.模型评估函数
"""
@:param  :   lstm_model  训练好的LSTM模型
@:param  :   train_x   训练数据
@:param  ：  train_y   训练数据对应值
@:param  ：  test_x    训练数据
@:param  ：  test_y    测试数据对应值
@:return :   predict   训练数据对应预测值
"""
def evaluate(lstm_model, train_x, train_y, test_x, test_y):
    print("train_x.shape=", train_x.shape)
    print("train_y.shape=", train_y.shape)
    print("test_x.shape=", test_x.shape)
    print("test_y.shape=", test_y.shape)

    pred_train = lstm_model.predict(train_x) # 训练样本
    pred_test = lstm_model.predict(test_x)   # 测试样本

    y_true = np.zeros((pred_train.shape[0], 2))
    tmp = np.zeros((pred_train.shape[0], 2))
    # 对预测数据反归一化
    y_true_inverse = sc.inverse_transform(np.concatenate((y_true, train_y.reshape(-1, 1)), axis=1))[:, -1]
    pred_inverse = sc.inverse_transform(np.concatenate((tmp, pred_train.reshape(-1, 1)), axis=1))[:, -1]

    y_true_test = np.zeros((pred_test.shape[0], 2))
    tmp_test = np.zeros((pred_test.shape[0], 2))
    # 对预测数据反归一化
    y_true_test_inverse = sc.inverse_transform(np.concatenate((y_true_test, test_y.reshape(-1, 1)), axis=1))[:, -1]
    pred_test_inverse = sc.inverse_transform(np.concatenate((tmp_test, pred_test.reshape(-1, 1)), axis=1))[:, -1]


    # 改变预测样本形状
    pred_train = np.reshape(pred_train, (pred_train.size,))
    pred_test = np.reshape(pred_test, (pred_test.size,))

    # 模型评估
    r2Score = r2_score(y_true=train_y, y_pred=pred_train)
    mse = mean_squared_error(y_true=train_y, y_pred=pred_train)
    mae = mean_absolute_error(y_true=train_y, y_pred=pred_train)
    print('训练样本评估指标：')
    print('r2 score: ', r2Score)
    print('mse: ', mse)
    print('mae: ', mae)

    r2Score = r2_score(y_true=test_y, y_pred=pred_test)
    mse = mean_squared_error(y_true=test_y, y_pred=pred_test)
    mae = mean_absolute_error(y_true=test_y, y_pred=pred_test)
    print('测试样本评估指标：')
    print('r2 score: ', r2Score)
    print('mse: ', mse)
    print('mae: ', mae)

    # 绘制真实值和预测值曲线（）
    fig = plt.figure(1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('LSTM拟合周一至周五全时间段出现需求')
    plt.savefig("./{} 拟合周一至周五全时间段出行需求.jpg".format('LSTM'))
    plt.ylabel('出现需求')
    plt.xlabel('时间段（小时）')

    plt.plot(range(len(pred_inverse)), pred_inverse, 'r', label='训练预测值')
    plt.plot(range(len(y_true_inverse)), y_true_inverse, 'b', label='训练真实值')
    plt.plot(range(len(pred_inverse), len(pred_inverse) + len(y_true_test_inverse)), y_true_test_inverse, 'g', label='测试预测值')
    plt.plot(range(len(y_true_inverse), len(y_true_inverse) + len(pred_test_inverse)), pred_test_inverse, 'y', label='测试真实值')

    plt.legend()

    return predict

# 4. 模型预测函数，对新样本进行预测
"""
@:param  :   saved_path  训练好的LSTM模型保存路径
@:param  :   data_new   新的用于预测的数据 
"""
def predict(saved_path, data_new):
    # 加载已经训练好的模型
    lstm_model = keras.models.load_model(saved_path)
    return lstm_model.predict(data_new)

# 主函数
if __name__ == '__main__':
    infile = "./data/demand"
    # 1.加载数据集
    train_x, train_y, test_x, test_y, sc = load_data(infile)
    # 2.训练模型
    lstm_model, saved_path = train_model(train_x, train_y, test_x, test_y)
    # 3.模型评估
    evaluate(lstm_model, train_x, train_y, test_x, test_y)
    # 4.预测新样本
    # 新样本预处理，即标准化
    data_new = train_x
    #sc.fit_transform(data_new)
    pred_new = predict(saved_path, data_new)

    y_pred = np.zeros((pred_new.shape[0], 2))
    # 对预测数据反归一化，得到真实值大小
    pred_inverse = sc.inverse_transform(np.concatenate((y_pred, pred_new.reshape(-1, 1)), axis=1))[:, -1]
    # 5.绘制新样本预测结果图
    plt.figure(2)
    plt.plot(range(len(pred_inverse)), pred_inverse, 'r', label='新样本预测值')
    plt.legend()

'''
训练样本评估指标：
r2 score:  0.8401075123472228
mse:  0.006967679845260529
mae:  0.05383338398827249
测试样本评估指标：
r2 score:  0.7648945746520328
mse:  0.008322383989131935
mae:  0.06785283991483462

结果分析：训练样本r2分数大于测试样本r2，存在过拟合问题。

'''