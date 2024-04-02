import numpy as np
import scipy.linalg as la

#考虑就用两个方向相同的信号分别做缺陷处理，在test函数中用MUSIC算法处理得到两个角度，对比；
#然后用神经网络对两种缺陷处理（TODO：可以参考其他论文进行缺陷处理）进行映射，然后分别用MUSIC算法处理信号1和映射后的信号2。
#用附加的环境噪声（TODO：增加多径的表示）表示到达角和离开角信号的时变差异。
#理想的情况是出来两个相似的bluekey多峰空间伪谱。
#TODO：对比加上对角线元素和不加对角线元素的区别

"""
def compare_diff(A, B):
     # 计算两个矩阵的特征值和特征向量
    eigvals_A, eigvecs_A = np.linalg.eig(A)
    eigvals_B, eigvecs_B = np.linalg.eig(B)

    # 特征值差异度
    eigval_diff = np.abs(eigvals_A - eigvals_B).mean()
    print("特征值差异度:", eigval_diff)

    # 特征向量的夹角余弦值
    # 对于每一对特征向量，计算它们的夹角余弦值
    cos_angles = []
    for i in range(len(eigvecs_A)):
        vec_A = eigvecs_A[:, i]
        vec_B = eigvecs_B[:, i]
        cos_angle = np.dot(vec_A, vec_B) / (np.linalg.norm(vec_A) * np.linalg.norm(vec_B))
        cos_angles.append(cos_angle)

    # 计算夹角余弦值的平均值
    mean_cos_angle = np.mean(cos_angles)
    print("特征向量夹角余弦值的平均值:", mean_cos_angle)
    return (A-B)/A
"""

def generate_sin_signal(N):
    # 参数设置
    f = 1 # 信号频率, Hz
    Fs = 10 * f # 采样率

    # 生成时间序列
    t = np.arange(N) / Fs

    # 生成正弦波信号作为I分量
    I = np.sin(2 * np.pi * f * t)

    # 生成余弦波信号作为Q分量
    Q = np.cos(2 * np.pi * f * t)

    # 合成复数信号
    signal_complex = I + 1j*Q

    return signal_complex

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

def generate_im2_conv_data(signals, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx, AP_mtx, pos_para, noise_flag):
    #这个不一定是需要同一个函数里完全出来，可以是反复利用不同的输入参数得到的，比如缺陷矩阵都是I的时候出来的就是无缺陷的情况
    im2_data = []
    #im2_data['multipath'] = []
    #im2_data['simu'] = []
    #TODO：这个地方可以在可行性部分放一个实验结果数据图，考虑校准/不考虑校准的角度互易结果，放cdf累计误差
    #TODO：对比线性网络和非线性网络的区别
    #MC_mtx, AP_mtx, pos_para = generate_random_defects(M, d, rho, mc_flag, ap_flag, pos_flag)

    for doa_idx in range(GRID_NUM):
        DOA = doa_min + grid * doa_idx
        for rep_idx in range(NUM_REPEAT):
            # TODO：这个地方noise的分布是对的吗？
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0
            signal_i = signals[doa_idx][rep_idx]
            #signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            # 这一步加入了传感器的位置误差（位置误差适用于发射和接收天线非同组的情况）
            array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
            phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
            a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
            # 这一步加入了幅度相位误差
            a_i = np.matmul(AP_mtx, a_i)
            # 这一步加入了天线耦合误差（天线耦合误差适用于发射和接收天线非同组的情况）
            a_i = np.matmul(MC_mtx, a_i)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i # 之所以用+而不是=是因为有多个信号的情况

            if noise_flag == True:
                array_output = array_signal + 1 * add_noise
            #array_output_nf = array_signal + 0 * add_noise  # noise-free output
            #array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
            array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
            
            """
            #取上三角，如果要包括对角线元素，删去索引中的+1即可
            cov_vector_nf_ = []
            cov_vector_ = []
            for row_idx in range(M):
                cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx):])
                cov_vector_.extend(array_covariance[row_idx, (row_idx):])
            """
            cov_vector_ = []
            for row_idx in range(M):
                cov_vector_.extend(array_covariance[row_idx, (row_idx):])
            #array_covariance = cov_vector_
            array_covariance = np.asarray(cov_vector_)
            #cov_vector_ = np.asarray(cov_vector_)
            arr_ext = np.dstack([array_covariance.real, array_covariance.imag])
            mean = np.mean(arr_ext)
            std = np.std(arr_ext)
            nor_arr_ext = (arr_ext - mean) / std
            #cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
            #cov_vector = np.around(cov_vector, 8)
            im2_data.append(nor_arr_ext)
    return im2_data

def generate_im2_conv_multi_data(signals, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx, AP_mtx, pos_para, noise_flag):
    im2_data = []
    #两个函数不一样的点仅仅在于这个函数在数据集里面生成包含2-3个多径的信号，问题就在于怎么生成这个多径呢？
    #首先这个多径一定是从input里面进来的，因为需要复用这个多径数据，所以我们单独开一个函数生成这个多径数组？
    #原本的数据是从-60到60一个角一个角生成出来的，多径要怎么遍历类似的情况呢？

    #先只考虑两条多径的情况，根据实验结果判断需不需要增加
    #两条多径，在多基础上写两层遍历即可
    signal_0 = signals[0][0][0]
    doa_max = doa_min + grid * GRID_NUM
    for doa_idx in range(GRID_NUM):
        doa1 = doa_min + grid * doa_idx
        DELTA_NUM = int((doa_max - doa1)/grid)
        print(doa_idx)
        for delta_idx in range(DELTA_NUM):
            doa2 = doa1 + grid * delta_idx
            DOAs = [doa1, doa2]
            for rep_idx in range(NUM_REPEAT):
                array_signal = 0
                for DOA in DOAs:
                    #signal_i = generate_sin_signal(N)
                    #signal_i = signal_i.reshape(1, -1)
                    #print("sin")
                    #print(signal_i.shape)
                    signal_i = 10 ** (SNR / 20) * signals[doa_idx][delta_idx][rep_idx]
                    #print("fix")
                    #signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                    #print("want")
                    #print(signal_i.shape)
                    # 这一步加入了传感器的位置误差（位置误差适用于发射和接收天线非同组的情况）
                    
                    array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
                    phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
                    a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
                    # 这一步加入了幅度相位误差
                    a_i = np.matmul(AP_mtx, a_i)
                    # 这一步加入了天线耦合误差（天线耦合误差适用于发射和接收天线非同组的情况）
                    a_i = np.matmul(MC_mtx, a_i)
                    array_signal_i = np.matmul(a_i, signal_0)
                    array_signal += array_signal_i # 之所以用+而不是=是因为有多个信号的情况

                # TODO：这个地方noise的分布是对的吗？
                add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                # noise的影响可能也比天线要大
                if noise_flag == True:
                    array_output = array_signal + 1 * add_noise
                #array_output_nf = array_signal + 0 * add_noise  # noise-free output
                #array_output = array_signal + 1 * add_noise
                array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
                
                """
                #取上三角，如果要包括对角线元素，删去索引中的+1即可
                cov_vector_nf_ = []
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx):])
                    cov_vector_.extend(array_covariance[row_idx, (row_idx):])
                """
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_.extend(array_covariance[row_idx, (row_idx):])
                #array_covariance = cov_vector_
                array_covariance = np.asarray(cov_vector_)

                #取实部虚部，归一化
                arr_ext = np.dstack([array_covariance.real, array_covariance.imag])
                mean = np.mean(arr_ext)
                std = np.std(arr_ext)
                nor_arr_ext = (arr_ext - mean) / std
                #cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                #cov_vector = np.around(cov_vector, 8)
                #print(nor_arr_ext.size)
                im2_data.append(nor_arr_ext)
    return im2_data

def generate_conv_test_data(signals, M, N, d, wavelength, DOA, SNR, MC_mtx, AP_mtx, pos_para, coh_flag):
    K = len(DOA)
    array_signal = 0
    signal_0 = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
    # 相干的信号源
    #signal_0 = generate_sin_signal(N)
    #signal_0 = signal_0.reshape(1, -1)
    #signal_0 = 10 ** (SNR[0] / 20) * signals[0]
    signal_0 = signals
    for ki in range(K):
        # 不相干的信号源
        if coh_flag == False:
            signal_i = signals[ki]
        #signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        #phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
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
    array_output = array_signal + add_noise
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

    cov_vector_ = []
    for row_idx in range(M):
        cov_vector_.extend(array_covariance[row_idx, (row_idx):])
    array_covariance = np.asarray(cov_vector_)
    #cov_vector_ = np.asarray(cov_vector_)
    arr_ext = np.dstack([array_covariance.real, array_covariance.imag])
    mean = np.mean(arr_ext)
    std = np.std(arr_ext)
    nor_arr_ext = (arr_ext - mean) / std
    #TODO：我怀疑这一步归一化有问题，感觉像是会影响相位的样子，这一步后续验证一下
    #TODO：检验数据归一化之前和数据归一化之后跑出来的空间谱是否是一致的
    #TODO：检查MUSIC算法到底为什么对误差这么敏感？单独列一个test.py，通过增减噪声来检验
    #TODO：整体还是写成encoder decoder的形式，可以考虑在两端都加入一层卷积（是否可以对称？）

    #cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
    #cov_vector = np.around(cov_vector, 8)
    #cov_vector = np.around(cov_vector, 8)
    #array_covariance = np.around(array_covariance, 8)
    #Note:取位数为8的情况统一等到最后画谱的时候再取
    return nor_arr_ext

def generate_spec_batches(data_input, data_output, batch_size, noise_flag):
    if noise_flag == 0:
        data_ = data_input['nf']
        label_ = data_output['nf']
    else:
        data_ = data_input['wn']
        label_ = data_output['wn']
    # label_ = data_train['target_spec']
    data_len = len(label_)

    # shuffle data, 注意这个地方shuffle得要一致
    shuffle_seq = np.random.permutation(range(data_len))
    data = [data_[idx] for idx in shuffle_seq]
    label = [label_[idx] for idx in shuffle_seq]

    # generate batches
    num_batch = int(data_len / batch_size)
    data_batches = []
    label_batches = []
    for batch_idx in range(num_batch):
        batch_start = batch_idx * batch_size
        batch_end = np.min([(batch_idx + 1) * batch_size, data_len])
        data_batch = data[batch_start : batch_end]
        label_batch = label[batch_start: batch_end]
        data_batches.append(data_batch)
        label_batches.append(label_batch)

    return np.array(data_batches), np.array(label_batches)

def reconstruct_covariance_matrix(upper_triangle_elements_, dimension):
    """根据上三角元素重构协方差矩阵"""
    upper_triangle_elements_ = np.squeeze(upper_triangle_elements_)
    upper_triangle_elements = []
    for i in range(upper_triangle_elements_.shape[0]):
        upper_triangle_elements.append(upper_triangle_elements_[i,0]+1j*upper_triangle_elements_[i,1])
    #upper_triangle_elements = np.around(upper_triangle_elements, 8)
    cov_matrix = np.zeros((dimension, dimension)) + 1j*np.zeros((dimension, dimension))
    cov_matrix[np.triu_indices(dimension)] = upper_triangle_elements
    cov_matrix = cov_matrix + cov_matrix.conj().T - np.diag(np.diag(cov_matrix))
    #print("recovered matrix")
    #print(cov_matrix)
    #cov2 = cov_matrix
    return cov_matrix

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

def spatial_smoothing_music(cov_mat, num_sources, angles):
    """
    空间平滑MUSIC算法实现
    :param cov_mat: 信号的协方差矩阵
    :param num_sources: 信号源数量
    :param array_size: 阵列元素总数
    :param subarray_size: 子阵列大小
    :return: MUSIC谱
    """

    array_size = cov_mat.shape[0]
    subarray_size = num_sources + 1
    num_subarrays = array_size - subarray_size + 1
    smoothed_cov_mat = np.zeros((subarray_size, subarray_size), dtype=complex)

    # 空间平滑
    for i in range(num_subarrays):
        subarray_cov_mat = cov_mat[i:i+subarray_size, i:i+subarray_size]
        smoothed_cov_mat += subarray_cov_mat

    smoothed_cov_mat /= num_subarrays

    # 特征分解
    eig_vals, eig_vecs = np.linalg.eigh(smoothed_cov_mat)
    idx = eig_vals.argsort()[::-1]   
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # 分离信号子空间和噪声子空间
    noise_subspace = eig_vecs[:, num_sources:]

    # MUSIC算法
    # angles = np.linspace(-90, 90, 360)
    music_spectrum = np.zeros(angles.shape)+1j*np.zeros(angles.shape)

    for i, angle in enumerate(angles):
        steering_vector = np.exp(-1j * np.pi * np.arange(subarray_size) * np.sin(np.deg2rad(-angle)))
        music_spectrum[i] = 1 / np.conj(steering_vector).dot(noise_subspace).dot(noise_subspace.conj().T).dot(steering_vector.T)

    music_spectrum = 10 * np.log10(music_spectrum.real)
    #P_music_dB = 10 * np.log10(P_music.real)  # 转换为dB
    return angles, music_spectrum

def generate_array_cov_vector(signals, M, N, d, wavelength, DOA, SNR, MC_mtx, AP_mtx, pos_para, co_flag):
    K = len(DOA)
    array_signal = 0
    #相干的信号源
    signal_0 = generate_sin_signal(N)
    #signal_0 = 10 ** (SNR[0] / 20) * signals[0]
    #signal_0 = 10 ** (SNR[0] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
    for ki in range(K):
        # 不相干的信号源
        signal_i = 10 ** (SNR[ki] / 20) * signals[ki]
        #signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
        array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
        phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
        a_i = np.matmul(AP_mtx, a_i)
        a_i = np.matmul(MC_mtx, a_i)
        if co_flag == True:
            array_signal_i = np.matmul(a_i, signal_0) # 原本是signal_i，如果是多径就用signal_0，如果是不相干的信号源就用signal_i
        else:
            array_signal_i = np.matmul(a_i, signal_i)
        array_signal += array_signal_i

    # TODO：这个地方可能需要更谨慎地生成噪声：分布；对每条路径一致吗？对每个信号源一致吗？
    add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
    #print(array_signal)
    #print(add_noise)
    array_output = array_signal + add_noise
    array_output = array_signal
    array_covariance = 1 / N * (np.dot(array_output, array_output.conj().T))
    arr_vector_ext = np.concatenate([array_covariance.real, array_covariance.imag])
    array_covariance = 1 / np.linalg.norm(arr_vector_ext) * array_covariance

    #print("original matrix")
    #print(array_covariance)
    #cov1 = array_covariance
    cov_vector_ = []
    for row_idx in range(M):
        cov_vector_.extend(array_covariance[row_idx, row_idx:])
    cov_vector_ = np.asarray(cov_vector_)
    cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
    cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
    #cov_vector = np.around(cov_vector, 8)
    #array_covariance = np.around(array_covariance, 8)
    #Note:取位数为8的情况统一等到最后画谱的时候再取
    return cov_vector, array_covariance

def generate_im2_multi_data(signals, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx, AP_mtx, pos_para):
    im2_data = {}
    im2_data['nf'] = []
    im2_data['wn'] = []
    #两个函数不一样的点仅仅在于这个函数在数据集里面生成包含2-3个多径的信号，问题就在于怎么生成这个多径呢？
    #首先这个多径一定是从input里面进来的，因为需要复用这个多径数据，所以我们单独开一个函数生成这个多径数组？
    #原本的数据是从-60到60一个角一个角生成出来的，多径要怎么遍历类似的情况呢？

    #先只考虑两条多径的情况，根据实验结果判断需不需要增加
    #两条多径，在多基础上写两层遍历即可
    doa_max = doa_min + grid * GRID_NUM
    for doa_idx in range(GRID_NUM):
        doa1 = doa_min + grid * doa_idx
        DELTA_NUM = int((doa_max - doa1)/grid)
        #print(doa_idx)
        for delta_idx in range(DELTA_NUM):
            doa2 = doa1 + grid * delta_idx
            DOAs = [doa1, doa2]
            for rep_idx in range(NUM_REPEAT):
                array_signal = 0
                for DOA in DOAs:
                    signal_i = signals[doa_idx][delta_idx][rep_idx]
                    # signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                    # 这一步加入了传感器的位置误差（位置误差适用于发射和接收天线非同组的情况）
                    array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
                    phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
                    a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
                    # 这一步加入了幅度相位误差
                    a_i = np.matmul(AP_mtx, a_i)
                    # 这一步加入了天线耦合误差（天线耦合误差适用于发射和接收天线非同组的情况）
                    a_i = np.matmul(MC_mtx, a_i)
                    array_signal_i = np.matmul(a_i, signal_i)
                    array_signal += array_signal_i # 之所以用+而不是=是因为有多个信号的情况

                # TODO：这个地方noise的分布是对的吗？
                add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                array_output_nf = array_signal + 0 * add_noise  # noise-free output
                array_output = array_signal + 1 * add_noise
                array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
                array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
                
                #取上三角，如果要包括对角线元素，删去索引中的+1即可
                cov_vector_nf_ = []
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx):])
                    cov_vector_.extend(array_covariance[row_idx, (row_idx):])
                
                #取实部虚部，归一化
                cov_vector_nf_ = np.asarray(cov_vector_nf_)
                cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
                cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
                #cov_vector_nf = np.around(cov_vector_nf, 8)
                im2_data['nf'].append(cov_vector_nf)

                cov_vector_ = np.asarray(cov_vector_)
                cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
                cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                #cov_vector = np.around(cov_vector, 8)
                im2_data['wn'].append(cov_vector)
    return im2_data

def generate_im2_data(signals, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT_SF, grid, GRID_NUM, MC_mtx, AP_mtx, pos_para):
    #这个不一定是需要同一个函数里完全出来，可以是反复利用不同的输入参数得到的，比如缺陷矩阵都是I的时候出来的就是无缺陷的情况
    im2_data = {}
    im2_data['nf'] = []
    im2_data['wn'] = []
    #im2_data['multipath'] = []
    #im2_data['simu'] = []
    #TODO：这个地方可以在可行性部分放一个实验结果数据图，考虑校准/不考虑校准的角度互易结果，放cdf累计误差
    #TODO：对比线性网络和非线性网络的区别
    #MC_mtx, AP_mtx, pos_para = generate_random_defects(M, d, rho, mc_flag, ap_flag, pos_flag)

    for doa_idx in range(GRID_NUM):
        DOA = doa_min + grid * doa_idx
        for rep_idx in range(NUM_REPEAT_SF):
            # TODO：这个地方noise的分布是对的吗？
            add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
            array_signal = 0
            signal_i = signals[doa_idx][rep_idx]
            #signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
            # 这一步加入了传感器的位置误差（位置误差适用于发射和接收天线非同组的情况）
            array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
            phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA / 180 * np.pi)
            a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
            # 这一步加入了幅度相位误差
            a_i = np.matmul(AP_mtx, a_i)
            # 这一步加入了天线耦合误差（天线耦合误差适用于发射和接收天线非同组的情况）
            a_i = np.matmul(MC_mtx, a_i)
            array_signal_i = np.matmul(a_i, signal_i)
            array_signal += array_signal_i # 之所以用+而不是=是因为有多个信号的情况

            array_output_nf = array_signal + 0 * add_noise  # noise-free output
            array_output = array_signal + 1 * add_noise
            array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
            array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
            
            #取上三角，如果要包括对角线元素，删去索引中的+1即可
            cov_vector_nf_ = []
            cov_vector_ = []
            for row_idx in range(M):
                cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx):])
                cov_vector_.extend(array_covariance[row_idx, (row_idx):])
            
            #取实部虚部，归一化
            cov_vector_nf_ = np.asarray(cov_vector_nf_)
            cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
            cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
            #cov_vector_nf = np.around(cov_vector_nf, 8)
            im2_data['nf'].append(cov_vector_nf)

            cov_vector_ = np.asarray(cov_vector_)
            cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
            cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
            #cov_vector = np.around(cov_vector, 8)
            im2_data['wn'].append(cov_vector)
            
            """
            # construct multi-task autoencoder target
            scope_label = int((DOA - doa_min) / SF_SCOPE)
            target_curr_pre = np.zeros([output_size * scope_label, 1])
            target_curr_post = np.zeros([output_size * (SF_NUM - scope_label - 1), 1])
            target_curr = np.expand_dims(cov_vector, axis=-1)
            target = np.concatenate([target_curr_pre, target_curr, target_curr_post], axis=0)
            data_train_sf['target_spec'].append(np.squeeze(target))
            """
    return im2_data

if __name__ == '__main__':
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
    grid = 1         # DOA step (degree) for generating different scenarios
    GRID_NUM = int((doa_max - doa_min) / grid)
    # SF_NUM = 6       # number of spatial filters
    # SF_SCOPE = (doa_max - doa_min) / SF_NUM   # spatial scope of each filter
    SNR = 10
    NUM_REPEAT = 1    # number of repeated sampling with random noise
    noise_flag = 1    # 0: noise-free; 1: noise-present

    # # array imperfection parameters
    mc_flag = True
    ap_flag = False
    pos_flag = True
    rho= 1
    MC_mtx1, AP_mtx1, pos_para1 = generate_random_defects(M, d, rho, mc_flag, ap_flag, pos_flag)
    
    inputs = generate_im2_conv_data(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx1, AP_mtx1, pos_para1)
    print(inputs[0])