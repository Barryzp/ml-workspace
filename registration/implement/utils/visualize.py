import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from .tools import Tools

# 可视化相关数据
class VisualizeData:
    # 初始化操作
    def __init__(self, moving, fixed, masked) -> None:
        self.moving_img = moving
        self.fixed_img = fixed
        self.masked_img = masked
        self.moving_height, self.moving_width = moving.shape
        self.fixed_height, self.fixed_width = fixed.shape
        self.mi_datas = None

    # delta代表在多大范围内进行的， interval代表的是步长
    def spawn_datas(self, delta, interval, csv_prefix = "visualize_data"):
        x_delta, y_delta = delta[0], delta[1]

        # 生成 x 和 y 的坐标点
        x = np.arange(0, x_delta+interval, interval)
        y = np.arange(0, y_delta+interval, interval)

        # 生成坐标点网格
        xx, yy = np.meshgrid(x, y)
        mi_datas = []

        threshold = [128, 180]
        # 以前的bound是：[0, 50], 原始版本
        bound = [0, 100]
        lamda_mis = 0.1

        offset = (self.fixed_width / 2, self.fixed_height / 2)

        coordinates = np.vstack([xx.ravel(), yy.ravel()]).T
        for i, coord in enumerate(coordinates):
            coordinate = coord.tolist()
            p = (coordinate[0] + offset[0], coordinate[1] + offset[1])

            size = (self.fixed_width, self.fixed_height)
            # Crop and rotate
            cropped_image = Tools.crop_rotate(self.moving_img, p, size, 0)
            cropped_image = np.uint8(np.array(cropped_image))
            mi = Tools.mutual_information(self.fixed_img, cropped_image)
            nmi = Tools.normalized_mutual_information(self.fixed_img, cropped_image)
            mi_masked_float = Tools.mutual_information(self.masked_img, cropped_image)
            sc = Tools.spatial_correlation(self.fixed_img, cropped_image, threshold, bound)
            sc_masked, diff_img, bse_with_masked, crop_with_masked = Tools.spatial_correlation_with_mask(self.masked_img, self.fixed_img, cropped_image, bound)
            sc_masked_masked, diff_img_, bse_with_masked_, crop_with_masked_ = Tools.spatial_correlation_with_mask(self.masked_img, self.masked_img, cropped_image, bound)

            mis = mi + lamda_mis * sc_masked
            mis_masked = mi_masked_float + lamda_mis * sc_masked_masked

            mi_item = p.copy()
            mi_item.append(mi)
            mi_item.append(mis)
            mi_item.append(sc)
            mi_item.append(sc_masked)
            mi_item.append(nmi)
            mi_item.append(mi_masked_float)
            mi_item.append(mis_masked)
            mi_datas.append(mi_item)

        # 将数据转换为 DataFrame
        mi_df = pd.DataFrame(mi_datas, columns=['x', 'y', 'mi', 'mis', 'sc', 'sc_masked', 'nmi', 'masked_mi', 'mis_masked'])
        # 将 DataFrame 保存为 CSV 文件
        mi_df.to_csv(f'visualize_datas/{csv_prefix}_mi_datas.csv', index=False)
        self.mi_datas = mi_df

        # azimuth = 45  # 方位角，水平旋转的度数
        # elevation = 45  # 仰角，垂直旋转的度数
    def show_datas(self, key, azimuth=45, elevation=45):
        x = self.mi_datas['x']
        y = self.mi_datas['y']
        z = self.mi_datas[key]
        # 创建网格
        X, Y = np.meshgrid(np.linspace(np.min(x), np.max(x), len(np.unique(x))),
                   np.linspace(np.min(y), np.max(y), len(np.unique(y))))
        Z = griddata((x, y), z, (X, Y), method='cubic')
        # 绘制三维曲面
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        # 添加坐标轴标签
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel(key)

        # 设置视角

        ax.view_init(elev=elevation, azim=azimuth)

        plt.show()