import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from .tools import Tools

# 实验工具
class ExpTools:

    def show_particle_accum_dist(diameter_hist):
        # 使用 numpy.histogram 划分数据并获取频数
        diameter_n, diameter_bins = np.histogram(diameter_hist, bins=10)
        temp = 0
        num_accum = []
        label_accum = []

        for i in range(len(diameter_n)):
            index = -(i+1)
            num = diameter_n[index]
            temp += num
            num_accum.append(temp)
            bin_left = diameter_bins[index]
            bin_right = diameter_bins[index - 1]
            label = f'{bin_left:.1f} - {bin_right:.1f}'
            label_accum.append(label)

        # 反转数组
        num_accum.reverse()
        plt.xticks([])
        # 计算每个 bin 的中心位置
        bin_centers = [(diameter_bins[i] + diameter_bins[i + 1]) / 2 for i in range(len(diameter_bins) - 1)]
        # 绘制直方图
        plt.bar(bin_centers, num_accum, width=5.0, edgecolor='black', align='center')
        for i in range(len(num_accum)):
            plt.text(bin_centers[i], -0.1, label_accum[-(i+1)], ha='center', va='top', rotation=45, fontsize=8, color='black')
            plt.text(bin_centers[i], num_accum[i], str(num_accum[i]), ha='center', va='bottom')

        # 添加标题和标签
        plt.title('Particle dist Histogram')

    # 展示粒径分布
    def show_particle_dist(diameter_hist, bins=10):
        # 绘制直方图
        n, bins, patches = plt.hist(diameter_hist, bins, color='blue', edgecolor='black')

        # 在每个 bin 上显示对应的数值
        # 在每个 bin 上显示对应的左右边界
        # 隐藏X轴的刻度

        plt.xticks([])
        for i in range(len(n)):
            bin_left = bins[i]
            bin_right = bins[i + 1]
            bin_center = (bin_left + bin_right) / 2
            label = f'{bin_left:.1f} - {bin_right:.1f}'
            plt.text(bin_center, -0.1, label, ha='center', va='top', rotation=45, fontsize=8, color='black')
             # 标注 bin 的数值
            plt.text(bin_center, n[i], str(int(n[i])), ha='center', va='bottom', fontsize=8, color='black')

        # 添加标题和标签
        plt.title('Histogram of Particle Size Distribution')
        # plt.xlabel('Particle Size') # x坐标
        plt.ylabel('Frequency')

    # 展示粒径分布图（1）：只需要输入粒径数组以及划分的区间即可
    def show_particle_hist_by_binsnum(diameter_hist, interval_num):
        ExpTools.show_particle_dist(diameter_hist, interval_num)

    # 展示粒径分布图（2）：只需要输入粒径数组以及粒径范围和最小区间间隔即可
    def show_particle_hist_by_bins(diameter_hist, diameter_range, step):
        min_val = diameter_range[0]
        max_val = diameter_range[1]
        # 生成固定间隔的数组
        bins = np.arange(min_val, max_val + step, step)
        ExpTools.show_particle_dist(diameter_hist, bins)