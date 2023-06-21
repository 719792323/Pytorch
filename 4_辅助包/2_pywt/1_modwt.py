import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data

if __name__ == '__main__':
    N = 1000
    # ecg = np.array([(10 if i % 2 == 0 else -10) for i in range(N)])
    t = np.linspace(1, 100, N)
    # ecg = 2_pywt.data.ecg()
    # ecg = np.sin(t) + np.cos(20 * t) + np.random.randn(N) * 0.1
    ecg = np.sin(t) * 5 + np.log2(t)*2 + np.random.randn(N)
    # 使用pywt库中的swt函数进行离散小波变换（论文中用的是db4）
    coeffs = pywt.swt(data=ecg, wavelet='db4', trim_approx=True, norm=True)
    # 提取近似系数（ca）和细节系数（details）
    """
    近似系数（ca）表示信号的低频信息和整体趋势。它捕捉到信号中的平滑变化和慢速变化的特征。近似系数对应于小波变换的低频分量。
    细节系数（details）表示信号的高频信息和细节。它捕捉到信号中的快速变化和细微的波动。细节系数对应于小波变换的高频分量。
    """
    ca = coeffs[0]
    details = coeffs[1:]

    # 打印原始ECG信号的方差
    print("Variance of the ecg signal = {}".format(np.var(ecg, ddof=1)))

    # 计算所有小波系数的方差之和，并打印结果
    variances = [np.var(c, ddof=1) for c in coeffs]
    detail_variances = variances[1:]
    print(variances)
    print("Sum of variance across all SWT coefficients = {}".format(
        np.sum(variances)))
    ylim = [ecg.min(), ecg.max()]

    # 绘制原始ECG信号和各个分解层级的小波系数图
    fig, axes = plt.subplots(len(coeffs) + 1)
    axes[0].set_title("normalized SWT decomposition")
    axes[0].plot(ecg)
    axes[0].set_ylabel('ECG Signal')
    axes[0].set_xlim(0, len(ecg) - 1)
    axes[0].set_ylim(ylim[0], ylim[1])

    for i, x in enumerate(coeffs):
        ax = axes[-i - 1]
        ax.plot(coeffs[i], 'g')
        if i == 0:
            ax.set_ylabel("A0")
        else:
            ax.set_ylabel("D%d" % (len(coeffs) - i))
        # Scale axes
        ax.set_xlim(0, len(ecg) - 1)
        ax.set_ylim(ylim[0], ylim[1])

    # 绘制细节系数的方差随分解层级的变化图
    level = np.arange(1, len(detail_variances) + 1)

    plt.figure(figsize=(8, 6))
    fontdict = dict(fontsize=16, fontweight='bold')
    plt.plot(level, detail_variances[::-1], 'k.')
    plt.xlabel("Decomposition level", fontdict=fontdict)
    plt.ylabel("Variance", fontdict=fontdict)
    plt.title("Variances of detail coefficients", fontdict=fontdict)
    plt.show()
