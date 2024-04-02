import numpy as np
import matplotlib.pyplot as plt

from im2_conv_model import *
from utils_im2 import *

import torch
from torch.utils.data import TensorDataset, DataLoader
import os

#测试一下MUSIC算法的误差容忍度

def generate_random_defects(M, d, rho, mc_flag, ap_flag, pos_flag):
    # mutual coupling matrix
    if mc_flag == True:
        mc_para = rho * 0.3 * np.exp(1j * 60 / 180 * np.pi)
        MC_coef = mc_para ** np.array(np.arange(M))
        MC_mtx = la.toeplitz(MC_coef)
    else:
        MC_mtx = np.identity(M)
    # amplitude & phase error
    if ap_flag == True:
        #TODO:这个地方可以写成是随机的
        #amp_coef = rho * np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.2])
        #phase_coef = rho * np.array([0.0, -30, -30, -30, -30, -30, 30, 30, 30, 30])
        amp_coef = rho * np.random.uniform(-0.5, 0.5, M)
        phase_coef = rho * np.random.uniform(-60, 60, M)
        AP_coef = [(1+amp_coef[idx])*np.exp(1j*phase_coef[idx]/180*np.pi) for idx in range(M)]
        AP_mtx = np.diag(AP_coef)
    else:
        AP_mtx = np.identity(M)
    # sensor position error
    if pos_flag == True:
        #pos_para_ = rho * np.array([0.0, -1, -1, -1, -1, -1, 1, 1, 1, 1]) * 0.2 * d
        pos_para_ =  rho * np.random.uniform(-0.5, 0.5, M) * d
        #pos_para_ = rho * np.array([0.0, -1, -2, -3, -2, -1, 2, 3, 2, 1]) * 0.2 *  d
        pos_para = np.expand_dims(pos_para_, axis=-1)
        #pos_para = pos_para[:M]
    else:
        pos_para = np.zeros([M, 1])
    return MC_mtx, AP_mtx, pos_para

def generate_test_data(signals, M, N, d, wavelength, DOA, SNR, MC_mtx, AP_mtx, pos_para, noise_flag, coh_flag, nor_flag=False):
    K = len(DOA)
    array_signal = 0
    
    signal_0 = signals[0]
    #signal_0 = 10 ** (SNR[0] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))

    # 如果信号是随机的，协方差矩阵当然也会引入随机性！GOD!怎么会在这个地方弄错了，救命啊
    # 相干的信号源
    for ki in range(K):
        signal_i = signals[ki]
        #signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # 不相干的信号源
        # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
        array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
        phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
        a_i = np.matmul(AP_mtx, a_i)
        a_i = np.matmul(MC_mtx, a_i)
        if coh_flag == True:
            array_signal_i = np.matmul(a_i, signal_0) # 原本是signal_i，如果是多径就用signal_0，如果是不相干的信号源就用signal_i
        else:
            array_signal_i = np.matmul(a_i, signal_i)
        array_signal += array_signal_i

    # TODO：这个地方可能需要更谨慎地生成噪声：分布；对每条路径一致吗？对每个信号源一致吗？
    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    #print(array_signal)
    #print(add_noise)
    if noise_flag == True:
        array_output = array_signal + add_noise
    else:
        array_output = array_signal
    array_covariance = 1 / N * (np.dot(array_output, array_output.conj().T))
    #print("original matrix")
    #print(array_covariance)
    #cov1 = array_covariance
    """
    cov_vector_ = []
    for row_idx in range(M):
        cov_vector_.extend(array_covariance[row_idx, row_idx:])
    cov_vector_ = np.asarray(cov_vector_)
    """
     #cov_vector_ = np.asarray(cov_vector_)
    arr_ext = np.dstack([array_covariance.real, array_covariance.imag])

    if nor_flag == True:
        mean = np.mean(arr_ext)
        std = np.std(arr_ext)
        arr_ext = (arr_ext - mean) / std
    #TODO：我怀疑这一步归一化有问题，感觉像是会影响相位的样子，这一步后续验证一下
    #TODO：检验数据归一化之前和数据归一化之后跑出来的空间谱是否是一致的
    #TODO：检查MUSIC算法到底为什么对误差这么敏感？单独列一个test.py，通过增减噪声来检验
    #TODO：整体还是写成encoder decoder的形式，可以考虑在两端都加入一层卷积（是否可以对称？）

    #cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
    #cov_vector = np.around(cov_vector, 8)
    #cov_vector = np.around(cov_vector, 8)
    #array_covariance = np.around(array_covariance, 8)
    #Note:取位数为8的情况统一等到最后画谱的时候再取
    return arr_ext

def MUSIC_algorithm(cov_matrix, signal_dimension, angles):
    dimension = cov_matrix.shape[0]
    #angles = np.linspace(-90, 90, 360)  # 角度范围
    """MUSIC算法实现"""
    # 特征分解
    eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
    # 按特征值大小排序（从大到小）
    idx = eig_vals.argsort()[::-1]
    eig_vecs = eig_vecs[:, idx]

    # 分离信号子空间和噪声子空间
    noise_subspace = eig_vecs[:, signal_dimension:]

    # 计算MUSIC谱
    P_music = np.zeros(len(angles))+1j*np.zeros(len(angles))
    for i, angle in enumerate(angles):
        steering_vector = np.exp(-1j * np.pi * np.arange(dimension) * np.sin(np.deg2rad(-angle)))
        P_music[i] = 1 / np.conj(steering_vector).dot(noise_subspace).dot(noise_subspace.conj().T).dot(steering_vector.T)

    P_music_dB = 10 * np.log10(P_music.real)  # 转换为dB
    return angles, P_music_dB

if __name__ == '__main__':
    # # array signal parameters
    fc = 2.4e9     # carrier frequency
    c = 3e8      # light speed
    #M是关键参数，可能需要调整和画图
    M = 4        # array sensor number
    N = 50       # snapshot number
    wavelength = c / fc  # signal wavelength
    d = 0.5 * wavelength  # inter-sensor distance

    # # im2im training parameters
    doa_min = -60      # minimal DOA (degree)
    doa_max = 60       # maximal DOA (degree)
    grid = 1         # DOA step (degree) for generating different scenarios
    GRID_NUM = int((doa_max - doa_min) / grid)
    # SF_NUM = 6       # number of spatial filters
    # SF_SCOPE = (doa_max - doa_min) / SF_NUM   # spatial scope of each filter
    SNR = 10
    NUM_REPEAT = 1    # number of repeated sampling with random noise

    test_DOA = np.array([-35, 55])
    test_K = len(test_DOA)
    test_SNR = np.array([10, 10])
    coh_flag = True
    smooth_flag= False
    nor_flag = False
    noise_flag = False     # 0: noise-free; 1: noise-present

    # # array imperfection parameters
    mc_flag = False
    ap_flag = False
    pos_flag = False 
    rho= 1
    MC_mtx1, AP_mtx1, pos_para1 = generate_random_defects(M, d, rho, mc_flag, ap_flag, pos_flag)

    signals_test = []
    for ki in range(test_K):
        signals_test.append((np.random.randn(1, N) + 1j * np.random.randn(1, N)))

    test_data = generate_test_data(signals_test, M, N, d, wavelength, test_DOA, test_SNR, MC_mtx1, AP_mtx1, pos_para1, noise_flag, coh_flag, nor_flag)
    #print(test_data.shape)
    test_data = test_data[:,:,0]+1j*test_data[:,:,1]

    print(test_data)
    angles = np.linspace(doa_min, doa_max, GRID_NUM)  # 角度范围
    
    sig_num = len(test_DOA)
    # 运行MUSIC算法
    if smooth_flag == True:
        angles, P_music_dB = spatial_smoothing_music(test_data, sig_num, angles)
    else:
        angles, P_music_dB = MUSIC_algorithm(test_data, sig_num, angles)
    plt.clf()
    # 绘制MUSIC谱
    plt.plot(angles, P_music_dB)
    plt.title('MUSIC Spectrum')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Spatial Spectrum (dB)')
    plt.grid(True)
    if nor_flag == True:
        plt.savefig('test_spectrum_nor.png', dpi=300)
    plt.savefig('test_spectrum.png', dpi=300)
