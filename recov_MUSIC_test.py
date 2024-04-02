import numpy as np
import matplotlib.pyplot as plt

#Note:针对MUSIC算法敏感性太强的问题，可以考虑多训练几位浮点数，最后只保留前面几位
def generate_array_cov_vector(M, N, d, wavelength, DOA, SNR, MC_mtx, AP_mtx, pos_para):
    K = len(DOA)
    array_signal = 0
    signal_0 = 10 ** (SNR[0] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
    # 相干的信号源
    for ki in range(K):
        signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # 不相干的信号源
        add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
        # TODO：这个地方可能需要更谨慎地生成噪声：分布；对每条路径一致吗？对每个信号源一致吗？
        # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
        array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d + pos_para
        phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
        a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
        a_i = np.matmul(AP_mtx, a_i)
        a_i = np.matmul(MC_mtx, a_i)
        array_signal_i = np.matmul(a_i, signal_0) # 原本是signal_i，如果是多径就用signal_0，如果是不相干的信号源就用signal_i
        array_signal += array_signal_i

    #print(array_signal)
    #print(add_noise)
    array_output = array_signal + add_noise
    array_output = array_signal
    array_covariance = 1 / N * (np.dot(array_output, array_output.conj().T))

    #print("original matrix")
    #print(array_covariance)
    #cov1 = array_covariance
    cov_vector_ = []
    for row_idx in range(M):
        cov_vector_.extend(array_covariance[row_idx, row_idx:])
    cov_vector_ = np.asarray(cov_vector_)
    cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
    cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
    array_covariance = 1 / np.linalg.norm(cov_vector_ext) * array_covariance
    cov_vector = np.around(cov_vector, 8)
    array_covariance = np.around(array_covariance, 8)
    return cov_vector, array_covariance

def reconstruct_covariance_matrix(upper_triangle_elements, dimension):
    """根据上三角元素重构协方差矩阵"""
    cov_matrix = np.zeros((dimension, dimension)) + 1j*np.zeros((dimension, dimension))
    cov_matrix[np.triu_indices(dimension)] = upper_triangle_elements
    cov_matrix = cov_matrix + cov_matrix.conj().T - np.diag(np.diag(cov_matrix))
    #print("recovered matrix")
    #print(cov_matrix)
    #cov2 = cov_matrix
    return cov_matrix

def MUSIC_algorithm(cov_matrix, signal_dimension, angles):
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

# 示例参数
dimension = 10  # 协方差矩阵的维度
#upper_triangle_elements = np.array([1, 0.9, 0.7, 0.6, 0.9, 0.8, 0.5, 0.4, 0.3, 0.2])  # 上三角元素示例

# # array signal parameters
fc = 2.4e9     # carrier frequency
c = 3e8      # light speed
#M是关键参数，可能需要调整和画图
M = dimension        # array sensor number
N = 50       # snapshot number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# # test data parameters
test_DOA = np.array([30, -45, 15])
test_K = len(test_DOA)
test_SNR = np.array([30, 20, 10])

signal_dimension = 1 # 信号源数量

MC_mtx = np.identity(M)
AP_mtx = np.identity(M)
pos_para = np.zeros([M, 1])

upper_triangle_elements_, array_covariance = generate_array_cov_vector(M, N, d, wavelength, test_DOA, test_SNR, MC_mtx, AP_mtx, pos_para)
cov1 = array_covariance
#print(cov1)
angles = np.linspace(-90, 90, 360)  # 角度范围
# 运行MUSIC算法
angles, P_music_dB = MUSIC_algorithm(array_covariance, signal_dimension, angles)

plt.clf()
# 绘制MUSIC谱
plt.plot(angles, P_music_dB)
plt.title('MUSIC Spectrum')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Spatial Spectrum (dB)')
plt.grid(True)
plt.savefig('MUSIC_Spectrum_origi.png', dpi=300)
#plt.show()
upper_triangle_elements = []
#print(upper_triangle_elements_)
ele_len = int(len(upper_triangle_elements_)/2)
for i in range(ele_len):
    upper_triangle_elements.append(upper_triangle_elements_[i]+1j*upper_triangle_elements_[i+ele_len])

# 重构协方差矩阵
cov_matrix = reconstruct_covariance_matrix(upper_triangle_elements, dimension)
cov2 = cov_matrix
#print(cov2)
angles = np.linspace(-90, 90, 360)  # 角度范围
# 运行MUSIC算法
angles, P_music_dB = MUSIC_algorithm(cov_matrix, signal_dimension, angles)

plt.clf()
# 绘制MUSIC谱
plt.plot(angles, P_music_dB)
plt.title('MUSIC Spectrum')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Spatial Spectrum (dB)')
plt.grid(True)
plt.savefig('MUSIC_Spectrum_recov.png', dpi=300)
#plt.show()

print("!!!get difference cov1 and cov2!!!")
print((cov1-cov2)/cov1)