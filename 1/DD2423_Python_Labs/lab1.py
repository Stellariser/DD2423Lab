import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import rotate
from Functions import showgrey, showfs  # 假设 showgrey 和 showfs 已定义
from numpy.fft import fft2, fftshift, ifft2

def display_frequency_properties():
    # 设置不同的频域坐标参数
    coordinates = [(5, 9), (9, 5), (17, 9), (17, 121), (5, 1), (125, 1)]
    sz = 128  # 图像尺寸

    f = plt.figure(figsize=(15, 18))
    f.subplots_adjust(wspace=0.4, hspace=0.6)

    for row, (u, v) in enumerate(coordinates):
        Fhat = np.zeros([sz, sz])
        Fhat[u, v] = 1
        F = np.fft.ifft2(Fhat)
        Fabsmax = np.max(np.abs(F))
        uc = u if u < sz / 2 else u - sz   #将大于图像尺寸一半的频率坐标平移到中心对称的负频率上，以便居中显示。
        vc = v if v < sz / 2 else v - sz
        wavelength = 1 / np.sqrt(uc**2 + vc**2)
        amplitude = 1.0 / sz * sz   # Fabsmax == 1

        f.add_subplot(6, 5, row * 5 + 1)
        showgrey(Fhat, False)
        plt.title(f"Fhat (u,v)=({u},{v})", fontsize=8)

        f.add_subplot(6, 5, row * 5 + 2)
        showgrey(np.fft.fftshift(Fhat), False)
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

    plt.show()

def display_images_and_spectra():
    F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T
    H = F + 2 * G

    Fhat = np.fft.fft2(F) # 2D Fourier Transform
    Ghat = np.fft.fft2(G)
    Hhat = np.fft.fft2(H)

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axs[0, 0].imshow(F, cmap='gray')
    axs[0, 0].set_title("Image F")

    axs[1, 0].imshow(G, cmap='gray')
    axs[1, 0].set_title("Image G")

    axs[2, 0].imshow(H, cmap='gray')
    axs[2, 0].set_title("Image H")

    axs[0, 1].imshow(np.log(1 + np.abs(Fhat)), cmap='gray')
    axs[0, 1].set_title("Fourier Spectrum of F")

    axs[1, 1].imshow(np.log(1 + np.abs(Ghat)), cmap='gray')
    axs[1, 1].set_title("Fourier Spectrum of G")

    axs[2, 1].imshow(np.log(1 + np.abs(Hhat)), cmap='gray')
    axs[2, 1].set_title("Fourier Spectrum of H")

    axs[0, 2].imshow(np.abs(np.fft.fftshift(Fhat)), cmap='gray') # no log
    axs[0, 2].set_title("Centered Fourier Spectrum of F")

    axs[1, 2].imshow(np.log(1 + np.abs(np.fft.fftshift(Ghat))), cmap='gray')
    axs[1, 2].set_title("Centered Fourier Spectrum of G")

    axs[2, 2].imshow(np.log(1 + np.abs(np.fft.fftshift(Hhat))), cmap='gray')
    axs[2, 2].set_title("Centered Fourier Spectrum of H")

    for ax in axs.flat:
        ax.axis('off')

    plt.show()

def display_convolution_in_frequency_domain():
    F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
    G = F.T

    f = plt.figure(1)
    f.subplots_adjust(wspace=0.2, hspace=0.4)

    Fhat = fft2(F)
    Ghat = fft2(G)

    a1 = f.add_subplot(1, 3, 1)
    showgrey(F * G, False)
    a1.title.set_text("F * G")

    a2 = f.add_subplot(1, 3, 2)
    showfs(np.fft.fft2(F * G), False)
    a2.title.set_text("fft(F * G)")

    a3 = f.add_subplot(1, 3, 3)
    showgrey(np.log(1 + np.abs(convolve2d(Fhat, Ghat, mode='same', boundary='wrap') / (128 ** 2))), False)
    a3.title.set_text("conv(Fhat, Ghat)")

    plt.show()

def display_scaled_image_and_spectrum():
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(wspace=0.4)

    axs[0].imshow(F, cmap='gray')
    axs[0].set_title("Scaled Image F")
    axs[0].axis('off')

    Fhat = np.fft.fft2(F)
    axs[1].imshow(np.log(1 + np.abs(fftshift(Fhat))), cmap='gray')
    axs[1].set_title("Fourier Spectrum of Scaled Image F")
    axs[1].axis('off')

    plt.show()

def display_rotated_image_and_spectrum():
    angles = [30, 45, 60, 90]
    F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
        np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)

    fig, axs = plt.subplots(len(angles), 4, figsize=(16, 4 * len(angles)))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, alpha in enumerate(angles):
        G = rotate(F, angle=alpha, reshape=False)
        Ghat = fft2(G)
        Hhat = rotate(fftshift(Ghat), angle=-alpha, reshape=False)

        axs[i, 0].imshow(G, cmap='gray')
        axs[i, 0].set_title(f"Rotated Image (alpha={alpha}°)")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(np.log(1 + np.abs(fftshift(Ghat))), cmap='gray')
        axs[i, 1].set_title(f"Fourier Spectrum of Rotated Image (alpha={alpha}°)")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(np.log(1 + np.abs(Hhat)), cmap='gray')
        axs[i, 2].set_title(f"Rotated Spectrum Back (alpha={-alpha}°)")
        axs[i, 2].axis('off')

        Fhat = fft2(F)
        axs[i, 3].imshow(np.log(1 + np.abs(fftshift(Fhat))), cmap='gray')
        axs[i, 3].set_title("Original Fourier Spectrum")
        axs[i, 3].axis('off')

    plt.show()

from Functions import pow2image, randphaseimage
import os

def analyze_fourier_phase_and_magnitude_all(a=1e-3):
    # 图像路径列表
    image_paths = [
        "Images-npy/phonecalc128.npy",
        "Images-npy/few128.npy",
        "Images-npy/nallo128.npy"
    ]


    for img_path in image_paths:
        if os.path.exists(img_path):
            img = np.load(img_path)

            # 替换功率谱后的图像
            img_pow2 = pow2image(img, a)

            # 随机化相位后的图像
            img_randphase = randphaseimage(img)

            # 显示原始图像、替换功率谱后的图像和随机相位处理后的图像
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].imshow(img, cmap='gray')
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            axs[1].imshow(img_pow2, cmap='gray')
            axs[1].set_title("Phrase")
            axs[1].axis('off')

            axs[2].imshow(img_randphase, cmap='gray')
            axs[2].set_title("Amplitude")
            axs[2].axis('off')

            plt.show()
        else:
            print(f"Image {img_path} not found.")

from Functions import deltafcn,variance
from gaussfft import gaussfft


def gauss_test():

    # 定义 t 值用于方差和模糊测试
    t_values = {
        "variance_test": [0.1, 0.3, 1.0, 10.0, 100.0],
        "blur_test": [1.0, 4.0, 16.0, 64.0, 256.0]
    }
    img = np.load("Images-npy/genevepark128.npy")

    # 脉冲响应和方差展示
    fig1, axs1 = plt.subplots(1, 5, figsize=(15, 3))
    fig1.subplots_adjust(wspace=0.3)
    plt.rc('axes', titlesize=10)

    for i, t in enumerate(t_values["variance_test"]):
        psf, X, Y, gauss = gaussfft(deltafcn(128, 128), t)
        var = variance(psf)
        var = [[round(j, 3) for j in var[i]] for i in range(len(var))]

        print(f"Variance for t={t}: {var:}")

        # 频域脉冲响应可视化
        ax_img = axs1[i]
        ax_img.imshow(psf, cmap='gray')
        ax_img.set_title(f't={t}\nvar={var:}')
        ax_img.axis('off')

    # 图像平滑效果展示
    fig2, axs2 = plt.subplots(1, 5, figsize=(15, 3))
    fig2.subplots_adjust(wspace=0.3)
    plt.rc('axes', titlesize=10)

    for i, t in enumerate(t_values["blur_test"]):
        blurred_img, _, _, _ = gaussfft(img, t)

        # 显示平滑后的图像
        ax = axs2[i]
        ax.imshow(blurred_img, cmap='gray')
        ax.set_title(f't={t}')
        ax.axis('off')

    plt.show()

from Functions import gaussnoise, sapnoise,  medfilt, ideal, rawsubsample
def smoothing():
    # 高斯平滑参数
    t_values = [0.1, 0.3, 0.5, 1.0, 2.0, 10.0]

    # 中值滤波参数
    window_sizes = [1, 3, 5, 7, 9, 11]

    # 理想低通滤波参数
    cutoff_frequencies = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    # 3.1 Smoothing of Noisy Data
    # 加载原始图像并添加噪声
    office = np.load("Images-npy/office256.npy")
    original = office.copy()
    add = gaussnoise(office, 16)  # 添加高斯噪声
    sap = sapnoise(office, 0.1, 255)  # 添加椒盐噪声

    # 显示原始和带噪声的图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(add, cmap='gray')
    axs[1].set_title("Gaussian Noise")
    axs[2].imshow(sap, cmap='gray')
    axs[2].set_title("Salt-and-Pepper Noise")
    plt.show()

    # 1. 高斯平滑 - 不同 t 值
    fig, axs = plt.subplots(2, len(t_values), figsize=(15, 6))
    fig.suptitle("Gaussian Smoothing with Different t Values")
    for i, t in enumerate(t_values):
        # 对带高斯噪声图像进行高斯平滑
        smoothed_add = gaussfft(add, t)[0]
        axs[0, i].imshow(smoothed_add, cmap='gray')
        axs[0, i].set_title(f"gaussnoise, t={t}")
        axs[0, i].axis('off')

        # 对带椒盐噪声图像进行高斯平滑
        smoothed_sap = gaussfft(sap, t)[0]
        axs[1, i].imshow(smoothed_sap, cmap='gray')
        axs[1, i].set_title(f"sapnoise, t={t}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # 2. 中值滤波 - 不同窗口大小
    fig, axs = plt.subplots(2, len(window_sizes), figsize=(15, 6))
    fig.suptitle("Median Filtering with Different Window Sizes")
    for i, w in enumerate(window_sizes):
        # 对带高斯噪声图像进行中值滤波
        smoothed_add = medfilt(add, w)
        axs[0, i].imshow(smoothed_add, cmap='gray')
        axs[0, i].set_title(f"gaussnoise, w={w}")
        axs[0, i].axis('off')

        # 对带椒盐噪声图像进行中值滤波
        smoothed_sap = medfilt(sap, w)
        axs[1, i].imshow(smoothed_sap, cmap='gray')
        axs[1, i].set_title(f"sapnoise, w={w}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # 3. 理想低通滤波 - 不同截止频率
    fig, axs = plt.subplots(2, len(cutoff_frequencies), figsize=(15, 6))
    fig.suptitle("Ideal Low-Pass Filtering with Different Cut-Off Frequencies")
    for i, c in enumerate(cutoff_frequencies):
        # 对带高斯噪声图像进行理想低通滤波
        smoothed_add = ideal(add, c)
        axs[0, i].imshow(smoothed_add, cmap='gray')
        axs[0, i].set_title(f"gaussnoise, cut-off={c}")
        axs[0, i].axis('off')

        # 对带椒盐噪声图像进行理想低通滤波
        smoothed_sap = ideal(sap, c)
        axs[1, i].imshow(smoothed_sap, cmap='gray')
        axs[1, i].set_title(f"sapnoise, cut-off={c}")
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def smoothing_and_subsampling():
    # 加载图像
    img = np.load("Images-npy/phonecalc256.npy")
    smoothimg_gaussian = img.copy()
    smoothimg_ideal = img.copy()
    N = 5  # 降采样次数

    # 设置高斯平滑参数和理想低通滤波器的截止频率
    gaussian_t = 4
    ideal_cutoff = 0.2

    # 创建可视化
    fig, axs = plt.subplots(3, N, figsize=(15, 8))
    fig.suptitle("Original, Gaussian Smoothed, and Ideal Low-Pass Smoothed Subsampling")

    for i in range(N):
        if i > 0:  # 对图像进行降采样
            img = rawsubsample(img)
            smoothimg_gaussian = gaussfft(smoothimg_gaussian, gaussian_t)[0]  # 高斯平滑
            smoothimg_gaussian = rawsubsample(smoothimg_gaussian)
            smoothimg_ideal = ideal(smoothimg_ideal, ideal_cutoff)  # 理想低通滤波
            smoothimg_ideal = rawsubsample(smoothimg_ideal)

        # 显示原始降采样图像
        axs[0, i].imshow(img, cmap='gray')
        axs[0, i].set_title(f"Original Subsampled {2 ** i}x")
        axs[0, i].axis('off')

        # 显示高斯平滑降采样图像
        axs[1, i].imshow(smoothimg_gaussian, cmap='gray')
        axs[1, i].set_title(f"Gaussian Smoothed {2 ** i}x")
        axs[1, i].axis('off')

        # 显示理想低通平滑降采样图像
        axs[2, i].imshow(smoothimg_ideal, cmap='gray')
        axs[2, i].set_title(f"Ideal Low-Pass Smoothed {2 ** i}x")
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.show()

# 主函数调用
if __name__ == "__main__":
    img_path = "Images-npy/phonecalc128.npy"
    # display_frequency_properties()   # Q1-6
    # display_images_and_spectra()  # Q7-9
    # display_convolution_in_frequency_domain()  # Q10
    # display_scaled_image_and_spectrum()  # Q11
    # display_rotated_image_and_spectrum()   # Q12
    # analyze_fourier_phase_and_magnitude_all(a=1e-3) # Q13
    # gauss_test() # Q14-16
    # smoothing()  #  Q17-18
    # smoothing_and_subsampling()