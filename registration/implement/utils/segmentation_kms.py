import cv2
import numpy as np
from sklearn.cluster import KMeans


class SegmentationKMS:
    def __init__(self) -> None:
        pass

    # 使用kmeans算法对图像进行分类, image是numpy对象
    def kmeans_image_segmentation(self, image, n_clusters=2):
        # 将图像转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 将图像数据转换为二维数组
        h, w = gray.shape[:2]
        img_array = gray.reshape((-1, 1))

        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(img_array)

        # 将聚类结果映射回原始图像尺寸
        segmented_image = np.zeros((h, w), dtype=np.uint8)
        for i in range(len(labels)):
            segmented_image[i // w, i % w] = labels[i] * (255 // (n_clusters - 1))

        return segmented_image
    
    # 对kms的结果进行形态学处理, kms_image是numpy对象
    def morphy_process_kms_image(self, kms_image, class_value=127, kernel_size=3):
        # 首先对图像进行二值化处理
        neg_cls = kms_image != class_value
        positive_cls = kms_image == class_value
        kms_image[neg_cls] = 0
        kms_image[positive_cls] = 255

        # 进行形态学的膨胀腐蚀等的操作
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # 膨胀操作
        dilated_image = cv2.dilate(kms_image, kernel, iterations=1)
        # 腐蚀操作
        out_image = cv2.erode(dilated_image, kernel, iterations=1)
        return out_image
    
    # 将形态学处理的结果选择性抹去，去掉那些较小的颗粒，保留那些较大的颗粒，从而减小配准时候的差异
    def filter_small_size_out(self, morphy_image, filter_size = 128, ):
        # 寻找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphy_image, 4, cv2.CV_32S)

        # 创建一个新的图像来存放结果
        new_image = np.zeros_like(morphy_image)

        # 遍历所有连通区域
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= filter_size:
                # 如果连通区域的大小大于阈值，则将其添加到新图像中
                new_image[labels == i] = 255

        return morphy_image
