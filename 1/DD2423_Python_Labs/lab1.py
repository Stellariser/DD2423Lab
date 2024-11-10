import numpy as np
import matplotlib.pyplot as plt
from Functions import showgrey  # 确保 showgrey 函数已定义并可用

# 设置不同的频域坐标参数
coordinates = [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]
sz = 128  # 图像尺寸

# 创建主图像窗口，大小设置为 6 行 5 列，每行展示一组不同参数的结果
f = plt.figure(figsize=(15, 18))
f.subplots_adjust(wspace=0.4, hspace=0.6)

# 遍历不同的 (u, v) 参数并展示各自的结果
for row, (u, v) in enumerate(coordinates):
    # 创建当前 (u, v) 参数下的频域图像
    Fhat = np.zeros([sz, sz])
    Fhat[u, v] = 1  # 在指定位置设置频域图像的值为 1

    # 计算逆傅里叶变换
    F = np.fft.ifft2(Fhat)
    Fabsmax = np.max(np.abs(F))  # 获取最大幅值

    # 计算中心化坐标
    uc = u if u < sz / 2 else u - sz
    vc = v if v < sz / 2 else v - sz
    wavelength = sz / np.sqrt(uc**2 + vc**2) if uc != 0 or vc != 0 else 0  # 波长计算
    amplitude = 1.0 / sz  # 振幅计算

    # 显示各部分图像
    f.add_subplot(6, 5, row * 5 + 1)
    showgrey(Fhat, False)  # 设置 display=False，推迟显示
    plt.title(f"Fhat (u,v)=({u},{v})", fontsize=8)

    f.add_subplot(6, 5, row * 5 + 2)
    showgrey(np.fft.fftshift(Fhat), False)  # 设置 display=False
    plt.title(f"Centered Fhat ({u},{v})", fontsize=8)

    f.add_subplot(6, 5, row * 5 + 3)
    showgrey(np.real(F), False, 64, -Fabsmax, Fabsmax)
    plt.title("Real Part", fontsize=8)

    f.add_subplot(6, 5, row * 5 + 4)
    showgrey(np.imag(F), False, 64, -Fabsmax, Fabsmax)
    plt.title("Imaginary Part", fontsize=8)

    f.add_subplot(6, 5, row * 5 + 5)
    showgrey(np.abs(F), False, 64, -Fabsmax, Fabsmax)
    plt.title(f"Magnitude\nAmp: {amplitude:.2e}\nWL: {wavelength:.2f}", fontsize=8)

# 一次性展示所有子图
plt.show()


# 定义图像
F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
G = F.T
H = F + 2 * G

# 计算傅里叶变换
Fhat = np.fft.fft2(F)
Ghat = np.fft.fft2(G)
Hhat = np.fft.fft2(H)

# 创建图像窗口，使用 3 行 3 列的网格布局
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# 显示原始图像 F、G、H
axs[0, 0].imshow(F, cmap='gray')
axs[0, 0].set_title("Image F")

axs[1, 0].imshow(G, cmap='gray')
axs[1, 0].set_title("Image G")

axs[2, 0].imshow(H, cmap='gray')
axs[2, 0].set_title("Image H")

# 显示 F、G、H 的傅里叶频谱
axs[0, 1].imshow(np.log(1 + np.abs(Fhat)), cmap='gray')
axs[0, 1].set_title("Fourier Spectrum of F")

axs[1, 1].imshow(np.log(1 + np.abs(Ghat)), cmap='gray')
axs[1, 1].set_title("Fourier Spectrum of G")

axs[2, 1].imshow(np.log(1 + np.abs(Hhat)), cmap='gray')
axs[2, 1].set_title("Fourier Spectrum of H")

# 使用 fftshift 显示中心化的傅里叶频谱
axs[0, 2].imshow(np.log(1 + np.abs(np.fft.fftshift(Fhat))), cmap='gray')
axs[0, 2].set_title("Centered Fourier Spectrum of F")

axs[1, 2].imshow(np.log(1 + np.abs(np.fft.fftshift(Ghat))), cmap='gray')
axs[1, 2].set_title("Centered Fourier Spectrum of G")

axs[2, 2].imshow(np.log(1 + np.abs(np.fft.fftshift(Hhat))), cmap='gray')
axs[2, 2].set_title("Centered Fourier Spectrum of H")

# 去掉多余的坐标轴
for ax in axs.flat:
    ax.axis('off')

plt.show()