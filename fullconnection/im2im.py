import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
#import scipy.linalg as la

from im2_model import *
from utils_im2 import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # 如果想要使用第一张GPU，只需设置 gpus[0]
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    print("only use gpu0")
    
    # 设置GPU内存增长选项，避免占用所有可用内存
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # 打印异常
    print(e)

# # array signal parameters
fc = 2.4e9     # carrier frequency
c = 3e8      # light speed
#M是关键参数，可能需要调整和画图
M = 10        # array sensor number
N = 50       # snapshot number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# # im2im training parameters
doa_min = -60      # minimal DOA (degree)
doa_max = 60       # maximal DOA (degree)
grid_sf = 1         # DOA step (degree) for generating different scenarios
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
# SF_NUM = 6       # number of spatial filters
# SF_SCOPE = (doa_max - doa_min) / SF_NUM   # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1    # number of repeated sampling with random noise

noise_flag_sf = 1    # 0: noise-free; 1: noise-present
amp_or_phase = 0   # show filter amplitude or phase: 0-amplitude; 1-phase

# # autoencoder parameters
input_size = M * (M+1) #TODO：考虑增加对角线元素
# hidden_size = int(1/2 * input_size) #本来是1/2
hidden_size = hidden_size_ss = [int(2/3* input_size), int(4/9* input_size), int(2/3* input_size)]
# 这个地方感觉得改成是对称的
output_size = input_size
batch_size = 32
num_epoch = 200
learning_rate = 0.001
# learning_rate_sf = 0.001

co_flag = True

"""
# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step_ss = 1         # DOA step (degree) for generating different scenarios
K_ss = 1            # signal number
# doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE   # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 10    # number of repeated sampling with random noise
# noise_flag_ss = 1    # 0: noise-free; 1: noise-present
"""

# # test data parameters
test_DOA = np.array([-25, 55, 35, 10])
test_K = len(test_DOA)
test_SNR = np.array([10, 10, 10, 10])

# # retrain the networks or not
reconstruct_nn_flag = True
retrain_im2_flag = True

# # file path of neural network parameters
model_path_nn = 'initial_model_im2.npy'
model_path_im2 = 'autoencoder_model_im2.npy'

"""
rmse_path = 'arrayimperf'
if mc_flag == True:
    rmse_path += '_mc'
if ap_flag == True:
    rmse_path += '_ap'
if pos_flag == True:
    rmse_path += '_pos'
rmse_path += '.npy'
"""

num_epoch_test = 1
Rho = np.arange(11) * 0.1
#for rho in Rho:

# # array imperfection parameters
mc_flag = True
ap_flag = False
pos_flag = True
rho= 1
MC_mtx1, AP_mtx1, pos_para1 = generate_random_defects(M, d, rho, mc_flag, ap_flag, pos_flag)

# # array imperfection parameters
mc_flag = False
ap_flag = True
pos_flag = False
rho= 1
MC_mtx2, AP_mtx2, pos_para2 = generate_random_defects(M, d, rho, mc_flag, ap_flag, pos_flag)

#TODO：rho从小到大，又是一个图

if reconstruct_nn_flag == True:
    tf.reset_default_graph()
    im2_0 = Im2_Model(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=output_size,
                      learning_rate=learning_rate,
                      reconstruct_nn_flag=True,
                      train_im2_flag=True,
                      model_path_nn=model_path_nn,
                      model_path_im2=model_path_im2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        var_dict_nn = {}
        for var in tf.trainable_variables():
            value = sess.run(var)
            var_dict_nn[var.name] = value
        np.save(model_path_nn, var_dict_nn)

if retrain_im2_flag == True:
    # generate spatial filter training dataset
    # TODO:重写utils中的数据生成函数
    # 注意这样的写法：如果要加多径需要考虑如何生成一致的多径矩阵

    if co_flag == True:
        data_train_input = generate_im2_multi_data(M, N, d, wavelength, SNR_sf, doa_min, NUM_REPEAT_SF, grid_sf, GRID_NUM_SF, MC_mtx1, AP_mtx1, pos_para1)
        data_train_output = generate_im2_multi_data(M, N, d, wavelength, SNR_sf, doa_min, NUM_REPEAT_SF, grid_sf, GRID_NUM_SF, MC_mtx2, AP_mtx2, pos_para2)
    else:
        data_train_input = generate_im2_data(M, N, d, wavelength, SNR_sf, doa_min, NUM_REPEAT_SF, grid_sf, GRID_NUM_SF, MC_mtx1, AP_mtx1, pos_para1)
        data_train_output = generate_im2_data(M, N, d, wavelength, SNR_sf, doa_min, NUM_REPEAT_SF, grid_sf, GRID_NUM_SF, MC_mtx2, AP_mtx2, pos_para2)
    #这个地方的数据是不对的，应该input和output分别生成一下。
    tf.reset_default_graph()
    im2_1 = Im2_Model(input_size=input_size,
                      hidden_size=hidden_size,
                      output_size=output_size,
                      learning_rate=learning_rate,
                      reconstruct_nn_flag=False,
                      train_im2_flag=True,
                      model_path_nn=model_path_nn,
                      model_path_im2=model_path_im2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        loss_values = []

        for epoch in range(num_epoch):
            [data_batches, label_batches] = generate_spec_batches(data_train_input, data_train_output, batch_size, noise_flag_sf)
            for batch_idx in range(len(data_batches)):
                data_batch = data_batches[batch_idx]
                label_batch = label_batches[batch_idx]
                feed_dict = {im2_1.data_train_: data_batch, im2_1.label_im2_: label_batch}
                _, loss = sess.run([im2_1.train_op_im2, im2_1.loss_im2], feed_dict=feed_dict)

                print('Epoch: {}, Batch: {}, loss: {:g}'.format(epoch, batch_idx, loss))
            loss_values.append(loss)

        var_dict_sf = {}
        for var in tf.trainable_variables():
            value = sess.run(var)
            var_dict_sf[var.name] = value
        np.save(model_path_im2, var_dict_sf)

        # 绘制loss随着epoch变化的图像
        plt.plot(loss_values)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.grid(True)

        # 将图像保存到文件
        plt.savefig('loss_vs_epoch.png')

#如果用autoencoder结构，考虑encoder对应上行缺陷链路，decoder对应下行缺陷链路，参考rfeat是怎么进行变量分离的，这样可以使得嵌入层的z为最终需要的隐含信息。
#需要注意的是，MUSIC算法利用的是天线信号之间的“相对信息”，这个相对信息应该如何计算（如果claim说阵列缺陷是非线性的），同一缺陷的中心设备对应不同缺陷的外围设备，网络需要重新训练吗？
#Note:这个地方需要在discussion中进行说明，校准设备和实际设备的收发链路不同为何不影响MUSIC算法的结果，最好能有仿真的结果来支撑这个结论。

# # test
tf.reset_default_graph()
im2_2 = Im2_Model(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    learning_rate=learning_rate,
                    reconstruct_nn_flag=False,
                    train_im2_flag=False,
                    model_path_nn=model_path_nn,
                    model_path_im2=model_path_im2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('testing...')

    smooth_flag = False
    sig_num = len(test_DOA)
    #sig_num = 1
    # # test 
    est_DOA = []
    MSE_rho = np.zeros([test_K, ])
    for epoch in range(num_epoch_test):
        #这一步test里面的数据都是带噪声的
        test_cov_vec1, _ = generate_array_cov_vector(M, N, d, wavelength, test_DOA, test_SNR, MC_mtx1, AP_mtx1, pos_para1, co_flag)
        test_cov_vec2, _ = generate_array_cov_vector(M, N, d, wavelength, test_DOA, test_SNR, MC_mtx2, AP_mtx2, pos_para2, co_flag)
        #缺陷参数需要存下来用来test的时候用。。。
        #或者从原来生成的数据里面取一部分出来当测试集
        data_batch = np.expand_dims(test_cov_vec1, axis=-1)
        #print(data_batch.shape)
        feed_dict = {im2_2.data_train_: np.transpose(data_batch)}
        #for attr in dir(enmod_3):
        #    print(attr, getattr(enmod_3, attr))
        im2_output = sess.run(im2_2.output_im2, feed_dict=feed_dict)
        #im2_output = np.transpose(im2_output)
        test_im2_output = np.squeeze(im2_output, axis=-1)

        cov0 = reconstruct_covariance_matrix(test_cov_vec1, M)
        cov1 = reconstruct_covariance_matrix(test_im2_output, M)
        cov2 = reconstruct_covariance_matrix(test_cov_vec2, M)

        #print(cov1)
        angles = np.linspace(doa_min, doa_max, GRID_NUM_SF)  # 角度范围
        
        # 运行MUSIC算法
        if smooth_flag == True:
            angles, P_music_dB = spatial_smoothing_music(cov0, sig_num, angles)
        else:
            angles, P_music_dB = MUSIC_algorithm(cov0, sig_num, angles)
        plt.clf()
        # 绘制MUSIC谱
        plt.plot(angles, P_music_dB)
        plt.title('MUSIC Spectrum')
        plt.xlabel('Angle (Degrees)')
        plt.ylabel('Spatial Spectrum (dB)')
        plt.grid(True)
        plt.savefig('MUSIC_Spectrum_input.png', dpi=300)

        if smooth_flag == True:
            angles, P_music_dB = spatial_smoothing_music(cov1, sig_num, angles) #len(test_DOA)
        else:
            angles, P_music_dB = MUSIC_algorithm(cov1, sig_num, angles) #len(test_DOA)
        plt.clf()
        # 绘制MUSIC谱
        plt.plot(angles, P_music_dB)
        plt.title('MUSIC Spectrum')
        plt.xlabel('Angle (Degrees)')
        plt.ylabel('Spatial Spectrum (dB)')
        plt.grid(True)
        plt.savefig('MUSIC_Spectrum_output.png', dpi=300)
        
        if smooth_flag == True:
            angles, P_music_dB = spatial_smoothing_music(cov2, sig_num, angles)
        else:
            angles, P_music_dB = MUSIC_algorithm(cov2, sig_num, angles)
        plt.clf()
        # 绘制MUSIC谱
        plt.plot(angles, P_music_dB)
        plt.title('MUSIC Spectrum')
        plt.xlabel('Angle (Degrees)')
        plt.ylabel('Spatial Spectrum (dB)')
        plt.grid(True)
        plt.savefig('MUSIC_Spectrum_label.png', dpi=300)

        #print("!!!两种缺陷的差异!!!")
        #diff2 = compare_diff(cov0, cov2)
        #print("!!!输出和标签的差异!!!")
        #diff1 = compare_diff(cov1, cov2)
        

        #TODO:在utils写一个复原协方差矩阵并进行MUSIC算法的操作用于评估最终效果
        #TODO：原test函数是直接用的两个信号源，应该将utils中的信号修改为多径相干的形式
        #可以先测试如果在单路径条件下训练是否可以推广到多径

        #ss_min = np.min(ss_output)
        #ss_output_regularized = [ss if ss > -ss_min else [0.0] for ss in ss_output]
        
        """
        plt.figure()
        x = np.arange(-60, 60, 1)
        plt.plot(x, ss_output_regularized)
        plt.savefig("test.png")
        est_DOA_ii = get_DOA_estimate(ss_output, test_DOA, doa_min, grid_ss)
        print(est_DOA_ii)
        est_DOA.append(est_DOA_ii)
        MSE_rho += np.square(est_DOA_ii - test_DOA)
        """
    #RMSE_rho = np.sqrt(MSE_rho / num_epoch_test)
        
    #TODO：增加缺陷的随机性
    #TODO：增加多径相干信号的讨论；不同源多信号可以处理，但是多径相干信号无法处理
    #TODO：考虑构造存在相干信号的数据集
    #目标是拉开input和output/label谱的差异度
    #TODO：考虑用smooth_MUSIC来处理多径相干信号