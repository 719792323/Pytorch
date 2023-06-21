# pip install dtale
from statsmodels.tsa.filters.hp_filter import hpfilter
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 100
    t = np.linspace(1, 10, N)
    ts = np.sin(t) + np.cos(20 * t) + np.random.randn(N) * 0.1
    plt.figure(figsize=(10, 12))
    for i, l in enumerate([0.1, 1, 10, 100, 1000, 10000]):
        plt.subplot(3, 2, i + 1)
        cycle, trend = hpfilter(ts, lamb=l)
        # original=cycle+trend
        plt.plot(ts, label='original')
        plt.plot(cycle, label="cycle")
        plt.plot(trend, label='trend')
        # 这两个加起来的曲线和original重叠
        # plt.plot(trend + cycle, label="cycle+trend")
        plt.legend()
        plt.title('$\lambda$=' + str(l))
    plt.show()
