import numpy as np
import matplotlib.pyplot as plt

from im2_conv_model import *
from utils_im2 import *

import torch
from torch.utils.data import TensorDataset, DataLoader
import os

# 列出所有可用的GPU设备
gpus = [i for i in range(torch.cuda.device_count())]
if gpus:
    # 选择要使用的GPU编号，这里我们选择第二张GPU（假设有多于一张GPU）
    chosen_gpu = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen_gpu)
    print("only use gpu1")  # 注意在PyTorch中索引是从0开始的，所以gpu1实际上是第二张GPU

    # 在PyTorch中，默认情况下GPU内存是动态分配的，不需要手动设置内存增长选项
    # 但如果你需要更精细的控制，可以考虑使用torch.cuda.set_per_process_memory_fraction()
    # 或者torch.cuda.set_per_process_memory_fraction()来限制内存使用

    # 设置当前设备，虽然通过CUDA_VISIBLE_DEVICES已经可以限定可见的GPU
    # 但是这个调用确保了如果有多个脚本或者进程，每个都能正确地选择各自的GPU
    torch.cuda.set_device(chosen_gpu)

#NOTE:对于相干情况，当train和test情况路径数量一样时，天线数量不影响精度，当路径数量不同时，天线数量越多效果越好
#TODO：现在感觉比较对了！现在需要1.解决一下模型加载的问题，以及2.处理一下损失函数的计算（在测试环节尽可能多使用一点图）
# # array signal parameters
fc = 2.4e9     # carrier frequency
c = 3e8      # light speed
#M是关键参数，可能需要调整和画图
M = 10        # array sensor number
N = 50       # snapshot number
wavelength = c / fc  # signal wavelength
d = 0.5 * wavelength  # inter-sensor distance

# # im2im training parameters
doa_min = -60     # minimal DOA (degree)
doa_max = 60       # maximal DOA (degree)
grid = 1         # DOA step (degree) for generating different scenarios
GRID_NUM = int((doa_max - doa_min) / grid)
# SF_NUM = 6       # number of spatial filters
# SF_SCOPE = (doa_max - doa_min) / SF_NUM   # spatial scope of each filter
SNR = 10
NUM_REPEAT = 1    # number of repeated sampling with random noise
noise_flag = 1    # 0: noise-free; 1: noise-present

batch_size = 32
num_epoch = 50
learning_rate = 0.001

coh_flag = True
smooth_flag = False
acc = 8

# # test data parameters
test_DOA = np.array([-35, 25, 55, 10])
#test_SNR = np.array([10, 10])

# # retrain the networks or not
retrain_im2_flag = False

# # file path of neural network parameters
model_path_im2 = 'trained_conv_im2.pth'
num_epoch_test = 1

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

model = CustomCNN(M)
criterion = nn.MSELoss()  # 假设我们的任务是回归，使用MSE作为损失函数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if retrain_im2_flag == True:
    signals_train = []
    if coh_flag == True:
        for doa_idx in range(GRID_NUM):
            signals_train.append([])
            doa1 = doa_min + grid * doa_idx
            DELTA_NUM = int((doa_max - doa1)/grid)
            #print(DELTA_NUM)
            for delta_idx in range(DELTA_NUM):
                signals_train[doa_idx].append([])
                for rep_idx in range(NUM_REPEAT):
                    #print(NUM_REPEAT)
                    signals_train[doa_idx][delta_idx].append(10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N)))
        data_train_input = generate_im2_conv_multi_data(signals_train, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx1, AP_mtx1, pos_para1, noise_flag)
        data_train_output = generate_im2_conv_multi_data(signals_train, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx2, AP_mtx2, pos_para2, noise_flag)

    else:
        for doa_idx in range(GRID_NUM):
            signals_train.append([])
            DOA = doa_min + grid * doa_idx
            for rep_idx in range(NUM_REPEAT):
                signals_train[doa_idx].append(10 ** (SNR / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N)))
        #print(np.array(signals).shape)
        data_train_input = generate_im2_conv_data(signals_train, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx1, AP_mtx1, pos_para1, noise_flag)
        data_train_output = generate_im2_conv_data(signals_train, M, N, d, wavelength, SNR, doa_min, NUM_REPEAT, grid, GRID_NUM, MC_mtx2, AP_mtx2, pos_para2, noise_flag)
    
    print((np.array(data_train_input).shape))
    print((np.array(data_train_output).shape))
    #inputs, targets = generate_spec_batches(data_train_input, data_train_output, batch_size, noise_flag)
    #NOTE：！！注意这个地方取一下8位小数，否则会出现精度问题！！

    data = torch.from_numpy(np.array(data_train_input)).float()
    labels = torch.from_numpy(np.array(data_train_output)).float()
    # 步骤2: 创建TensorDataset和DataLoader
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 训练过程
    print("start training")
    # 记录损失值的列表
    losses = []
    for epoch in range(num_epoch):
        model.train() # 设置模型为训练模式
        epoch_losses = []
        for batch_idx, (data, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)  # 确保labels的数据类型和outputs匹配
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epoch}, Loss: {epoch_loss}')

    # 保存模型参数
    torch.save(model.state_dict(), model_path_im2)
    #model.save_parameters(model_path_im2)

    #print("Loss after training:", losses[-1])
    # 绘制损失图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epoch+1), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('conv_training_loss.png')  # 保存损失图为图片
    #plt.show()  # 显示图表

else:
    # 加载模型参数
    model.load_state_dict(torch.load(model_path_im2))

# 测试步骤（这里简化，只打印损失）
# TODO：先去把utils里面的数据都正则归一化！！！
# TODO：重新训练 
# TODO：把test环节的数据重新写一下！

#sig_num = len(test_DOA)

if coh_flag == True:
    sig_num = 1
else:
    sig_num = len(test_DOA)


model.eval()  # 设置模型为评估模式
with torch.no_grad():

    
    if coh_flag == True:
        signals_test = np.random.randn(1, N) + 1j * np.random.randn(1, N)
    else:
        signals_test = []
        for ki in range(len(test_DOA)):
            signals_test.append((np.random.randn(1, N) + 1j * np.random.randn(1, N)))
    inputs_ = generate_conv_test_data(signals_test, M, N, d, wavelength, test_DOA, SNR, MC_mtx1, AP_mtx1, pos_para1, coh_flag)
    targets_ = generate_conv_test_data(signals_test, M, N, d, wavelength, test_DOA, SNR, MC_mtx2, AP_mtx2, pos_para2, coh_flag)

    print(inputs_.shape)
    
    input_tensor = torch.from_numpy(inputs_).float().unsqueeze(0)
    target_tensor = torch.from_numpy(targets_).float().unsqueeze(0)
    output_tensor = model(input_tensor)
    loss_test = criterion(output_tensor, target_tensor)
    print("Loss of evaluation", loss_test.item())
    outputs_ = output_tensor.squeeze(0).numpy()

    #NOTE：这个地方精度取多少需要斟酌一下
    inputs_ = np.around(inputs_, acc)
    outputs_ = np.around(outputs_, acc)
    targets_ = np.around(targets_, acc)

    """
    input_mat = inputs_[:,:,0]+1j*inputs_[:,:,1]
    output_mat = outputs_[:,:,0]+1j*outputs_[:,:,1]
    target_mat = targets_[:,:,0]+1j*targets_[:,:,1]
    """

    input_mat = reconstruct_covariance_matrix(inputs_, M)
    output_mat = reconstruct_covariance_matrix(outputs_, M)
    target_mat = reconstruct_covariance_matrix(targets_, M)

    #print(cov1)
    angles = np.linspace(doa_min, doa_max, GRID_NUM)  # 角度范围
    # 运行MUSIC算法
    if smooth_flag == True:
        angles, P_music_dB = spatial_smoothing_music(input_mat, sig_num, angles)
    else:
        angles, P_music_dB = MUSIC_algorithm(input_mat, sig_num, angles)
    
    plt.clf()
    # 绘制MUSIC谱
    plt.plot(angles, P_music_dB)
    plt.title('MUSIC Spectrum')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Spatial Spectrum (dB)')
    plt.grid(True)
    plt.savefig('conv_MUSIC_Spectrum_input.png', dpi=300)

    if smooth_flag == True:
        angles, P_music_dB = spatial_smoothing_music(output_mat, sig_num, angles) #len(test_DOA)
    else:
        angles, P_music_dB = MUSIC_algorithm(output_mat, sig_num, angles) #len(test_DOA)
    plt.clf()
    # 绘制MUSIC谱
    plt.plot(angles, P_music_dB)
    plt.title('MUSIC Spectrum')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Spatial Spectrum (dB)')
    plt.grid(True)
    plt.savefig('conv_MUSIC_Spectrum_output.png', dpi=300)
    
    if smooth_flag == True:
        angles, P_music_dB = spatial_smoothing_music(target_mat, sig_num, angles)
    else:
        angles, P_music_dB = MUSIC_algorithm(target_mat, sig_num, angles)
    plt.clf()
    # 绘制MUSIC谱
    plt.plot(angles, P_music_dB)
    plt.title('MUSIC Spectrum')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Spatial Spectrum (dB)')
    plt.grid(True)
    plt.savefig('conv_MUSIC_Spectrum_label.png', dpi=300)