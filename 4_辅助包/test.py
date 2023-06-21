import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data
from statsmodels.tsa.filters.hp_filter import hpfilter

if __name__ == '__main__':
    N = 100
    t = np.linspace(1, 10, N)
    ecg = np.sin(t) * 1 + np.log2(t) * 10 + np.random.randn(N)
    l = 100
    cycle, trend = hpfilter(ecg, lamb=l)
    plt.plot(ecg, label='original')
    plt.plot(cycle, label="cycle")
    plt.plot(trend, label='trend')
    plt.legend(["original", "cycle", "trend"])
    plt.show()

    ecg = cycle

    coeffs = pywt.swt(data=ecg, wavelet='db8', trim_approx=True, norm=True)

    ca = coeffs[0]
    details = coeffs[1:]

    # 计算所有小波系数的方差之和，并打印结果
    variances = [np.var(c, ddof=1) for c in coeffs]
    detail_variances = variances[1:]
    print(variances)
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
