import cv2
import numpy as np
from utils.tools import Tools
from sklearn.cluster import KMeans


class SegmentationKMS:
    def __init__(self) -> None:
        pass

    # 使用kmeans算法对图像进行分类, image是numpy对象
    def kmeans_image_segmentation(self, image, n_clusters=2, random = None):
        # 将图像转换为灰度图
        gray = image #cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将图像数据转换为二维数组
        h, w = gray.shape[:2]
        img_array = gray.reshape((-1, 1))

        rand_seed = None
        if random != None : rand_seed = random

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=rand_seed)
        labels = kmeans.fit_predict(img_array)

        # 将聚类结果映射回原始图像尺寸
        segmented_image = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(labels)):
            segmented_image[i // w, i % w] = labels[i] * (255 // (n_clusters - 1))

        return segmented_image
    
    # 对kms的结果进行形态学处理, kms_image是numpy对象
    def morphy_process_kms_image(self, kms_image, kernel_size=3):

        # 进行形态学的膨胀腐蚀等的操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 膨胀操作
        dilated_image = cv2.dilate(kms_image, kernel, iterations=1)
        # 腐蚀操作
        out_image = cv2.erode(dilated_image, kernel, iterations=1)
        return out_image
    
    # 将形态学处理的结果选择性抹去，去掉那些较小的颗粒，保留那些较大的颗粒，从而减小配准时候的差异
    def filter_small_size_out(self, morphy_image, quantile = 20):
        # 对轮廓进行提取
        contour_img, contours = Tools.find_contours_in_bin_img(morphy_image)
        filled_img = Tools.fill_contours(contour_img, contours)

        dist_diameter_min, dist_diameter_max = Tools.get_typical_particle_diameter(contours, quantile)

        filtered_img = Tools.filter_by_diameter(contours, filled_img, [dist_diameter_min, dist_diameter_max])

        return filtered_img, [dist_diameter_min, dist_diameter_max]