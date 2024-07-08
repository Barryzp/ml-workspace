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

    def show_image(image, size = (15, 15), gray=True):
        plt.figure(figsize=size)
        if gray : plt.imshow(image, cmap='gray',vmin=0, vmax=255)
        else : plt.imshow(image)
        plt.axis('off')
        plt.show()

    def show_imgs_rgb(imgs, imgs_name, cols = 2):
        count = len(imgs)
        rows = count // cols + 1
        plt.figure(figsize=(12, 8))
        for i in range(count):
            plt.subplot(rows, cols, i + 1), plt.imshow(imgs[i]), plt.title(f'{imgs_name[i]}')
        # 调整间距
        plt.subplots_adjust(hspace=0.5, wspace=0.5)  # hspace控制垂直间距, wspace控制水平间距
        # 使用tight_layout自动调整子图参数
        plt.tight_layout()
        plt.show()

    # 展示图像（灰度）
    def show_imgs(imgs, imgs_name, cols = 2, gray = None):
        count = len(imgs)

        if gray is None:
            gray = [True] * count

        rows = count // cols + 1
        plt.figure(figsize=(12, 8))
        for i in range(count):
            plt.subplot(rows, cols, i + 1)
            if gray[i] : plt.imshow(imgs[i], cmap='gray',vmin=0, vmax=255)
            else: plt.imshow(imgs[i])
            plt.title(f'{imgs_name[i]}')
        # 调整间距
        plt.subplots_adjust(hspace=0.5, wspace=0.5)  # hspace控制垂直间距, wspace控制水平间距
        # 使用tight_layout自动调整子图参数
        plt.tight_layout()
        plt.show()

    def concate_imgs_show(imgs, cols, space, size = (15, 15), gray = True):
        count = len(imgs)
        rows = count // cols

        width, height = imgs[0].shape[1], imgs[0].shape[0]

        vertical_space = np.zeros([width, space])
        # 先进行水平拼接
        hori_concates = None
        # 每一行把图像全部拼起来
        for i in range(rows):
            cocates = None
            for j in range(cols):
                id = i * cols + j
                if j < cols - 1:
                    if cocates is None: 
                        cocates = np.concatenate([imgs[id], vertical_space], axis=1)
                    else: 
                        cocates = np.concatenate([cocates, imgs[id], vertical_space], axis=1)
                else:
                    cocates = np.concatenate([cocates, imgs[id]], axis=1)
            horizontal_space = np.zeros([space, cocates.shape[1]])
            if i < rows - 1:
                if hori_concates is None: hori_concates = np.concatenate([cocates, horizontal_space], axis=0)
                else: hori_concates = np.concatenate([hori_concates, cocates, horizontal_space], axis=0)
            else: hori_concates = np.concatenate([hori_concates, cocates], axis=0)

        VisualizeData.show_image(hori_concates, size, gray)

    def show_hist(hist_array, bins = 10, title = "histgram", color = 'blue', edgecolor = 'black', show_x = True, show_y = True, x_label = None, y_label = None):
        # 绘制直方图
        n, bins, patches = plt.hist(hist_array, bins=bins, color=color, edgecolor=edgecolor)
        # 在每个 bin 上显示对应的数值
        # 在每个 bin 上显示对应的左右边界
        # 隐藏X轴的刻度
        if not show_y : plt.yticks([])
        if not show_x : plt.xticks([])
        for i in range(len(n)):
            bin_left = bins[i]
            bin_right = bins[i + 1]
            bin_center = (bin_left + bin_right) / 2
            label = f'{bin_left:.1f} - {bin_right:.1f}'
            plt.text(bin_center, -0.1, label, ha='center', va='top', rotation=45, fontsize=8, color='black')

        # 添加标题和标签
        if title is not None : plt.title('Histogram of Particle Size Distribution')
        if x_label is not None : plt.xlabel('Particle Size')
        if y_label is not None : plt.ylabel('Frequency')


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