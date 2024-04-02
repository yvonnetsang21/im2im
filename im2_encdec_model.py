import numpy as np

def generate_im2_conv_multi_data(M, N, d, wavelength, SNR, doa_min, NUM_REPEAT_SF, grid, GRID_NUM, MC_mtx, AP_mtx, pos_para):
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
        print(doa_idx)
        for delta_idx in range(DELTA_NUM):
            doa2 = doa1 + grid * delta_idx
            DOAs = [doa1, doa2]
            for rep_idx in range(NUM_REPEAT_SF):
                array_signal = 0
                for DOA in DOAs:
                    signal_i = 10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
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
                
                """
                #取上三角，如果要包括对角线元素，删去索引中的+1即可
                cov_vector_nf_ = []
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx):])
                    cov_vector_.extend(array_covariance[row_idx, (row_idx):])
                """

                #取实部虚部，归一化
                #cov_vector_nf_ = np.asarray(cov_vector_nf_)
                arr_nf_ext = np.dstack([array_covariance_nf.real, array_covariance_nf.imag])
                #print("!!!!!here!!!!!")
                #print(cov_vector_nf_ext.shape)
                mean = np.mean(arr_nf_ext)
                std = np.std(arr_nf_ext)
                nor_arr_nf_ext = (arr_nf_ext - mean) / std
                #cov_vector_nf = np.around(cov_vector_nf, 8)
                im2_data['nf'].append(nor_arr_nf_ext)

                #cov_vector_ = np.asarray(cov_vector_)
                arr_ext = np.dstack([array_covariance.real, array_covariance.imag])
                mean = np.mean(arr_ext)
                std = np.std(arr_ext)
                nor_arr_ext = (arr_ext - mean) / std
                #cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                #cov_vector = np.around(cov_vector, 8)
                im2_data['wn'].append(nor_arr_ext)
    return im2_data['wn']

#if __name__ == '__main__':