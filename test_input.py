import numpy as np
import matplotlib.pyplot as plt

# 参数设置
f = 1 # 信号频率, Hz
N = 100 # 样本数
Fs = 10 * f # 采样率

# 生成时间序列
t = np.arange(N) / Fs

# 生成正弦波信号作为I分量
I = np.sin(2 * np.pi * f * t)

# 生成余弦波信号作为Q分量
Q = np.cos(2 * np.pi * f * t)

# 合成复数信号
signal_complex = I + 1j*Q

print(signal_complex.shape)

"""
# 绘图显示I和Q分量
plt.figure(figsize=(14, 6))

# 绘制I分量
plt.subplot(2, 1, 1)
plt.plot(t, I, label='I (In-phase)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('I (In-phase) Component')
plt.grid(True)
plt.legend()

# 绘制Q分量
plt.subplot(2, 1, 2)
plt.plot(t, Q, label='Q (Quadrature)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Q (Quadrature) Component')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("sinwave.png")
"""
