# Main function of PJ
# Author: Liu Jiayan
# E-mail: 21210720185@m.fudan.edu.cn
import os

import numpy as np
import scipy.io as scio
from train_Emg import train_pattern1, predict_pattern1, train_pattern2,predict_pattern2
import time

pattern = 1  # 1 or 2
path_data = 'E:/研一/研二/机器学习_助教/PJ/data_emg/'  # the path you save data_emg(end with '/')
subject = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
           '19', '20']
acc = np.zeros([len(subject), 2])
for i in range(len(subject)):
    for j in range(2):
        print(i, j)
        path_tmp = path_data + 'subject' + subject[i] + '_session' + str(j + 1)
        if not os.path.exists(path_tmp):
            continue
        filename = path_tmp + '/sample_train_motion.mat'
        file = scio.loadmat(filename)
        sample_train_motion = file['sample_train_motion'][0]

        filename = path_tmp + '/sample_train_rest.mat'
        file = scio.loadmat(filename)
        sample_train_rest = file['sample_train_rest'][0]

        filename = path_tmp + '/sample_validation_motion.mat'
        file = scio.loadmat(filename)
        sample_validation_motion = file['sample_validation_motion'][0]

        filename = path_tmp + '/sample_validation_rest.mat'
        file = scio.loadmat(filename)
        sample_validation_rest = file['sample_validation_rest'][0]

        filename = path_tmp + '/label_train.mat'
        file = scio.loadmat(filename)
        label_train = file['label_train'][0]

        filename = path_tmp + '/label_validation.mat'
        file = scio.loadmat(filename)
        label_validation = file['label_validation'][0]

        if pattern == 1:
            mdl = train_pattern1(sample_train_motion, label_train)
            label_predict = predict_pattern1(mdl, sample_validation_motion)
            acc_motion = sum(label_predict == label_validation) / len(label_validation)
            acc[i, j] = acc_motion
        elif pattern == 2:
            mdl = train_pattern2(sample_train_motion, sample_train_rest, label_train)
            label_predict_rest = predict_pattern2(mdl, sample_validation_rest)
            label_predict_motion = predict_pattern2(mdl, sample_validation_motion)
            acc_motion = sum(label_predict_motion == label_validation) / len(label_predict_motion)
            acc_rest = sum(label_predict_rest == np.repeat(11, len(label_predict_rest), axis=0)) / len(label_predict_motion)
            acc[i, j] = (acc_motion + acc_rest) / 2