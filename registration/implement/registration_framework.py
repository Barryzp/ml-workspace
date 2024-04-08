import numpy as np
import torch
import cv2
import itk

from utils.tools import Tools


class Registration:
    def __init__(self, config) -> None:
        self.itk_img = None
        self.ref_img = None
        self.pso_framework = None
        self.config = config

    def _load_moving_img(self):
        data_path = self.config.data_path
        cement_sample_index = self.config.cement_sample_index
        sample_bse_index = self.config.sample_bse_index
        ct_2d_index = self.config.ct_2d_index

        ct_image_path = f"{data_path}/sample{cement_sample_index}/ct/s{sample_bse_index}/enhanced"
        self.moving_image = cv2.imread(f"{ct_image_path}/slice_enhanced_{ct_2d_index}.bmp", cv2.IMREAD_GRAYSCALE)
    
    def get_moving_img_shape(self):
        # (height, width)(rows, column)
        return self.moving_image.shape

    # 加载itk图像序列
    def _load_itk(self):
        data_path = self.config.data_path
        cement_sample_index = self.config.cement_sample_index
        sample_bse_index = self.config.sample_bse_index
        start_index = self.config.start_index
        end_index = self.config.end_index
        count = end_index - start_index + 1

        ct_image_path = f"{data_path}/sample{cement_sample_index}/ct/s{sample_bse_index}/enhanced"

        ct_image1 = cv2.imread(f"{ct_image_path}/slice_enhanced_10.bmp", cv2.IMREAD_GRAYSCALE)
        ct_image_filenames = [f"{ct_image_path}/slice_enhanced_{start_index + i}.bmp" for i in range(count)]
        image_shape = (len(ct_image_filenames), ct_image1.shape[0], ct_image1.shape[1])

        image_array = np.zeros(image_shape, dtype=np.uint8)
        for i, filename in enumerate(ct_image_filenames):
            image_ = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image_array[i] = image_

        self.itk_img = itk.image_from_array(image_array)
    
    # 加载参考图像
    def _load_ref_img(self):
        data_path = self.config.data_path
        bse_zoom_times = self.config.bse_zoom_times
        cement_sample_index = self.config.cement_sample_index

        # HACK 测试用例，先测试这个小块儿能不能跑通
        bse_file_name = "4-1-1-enhanced-roi-300"
        bse_file_path = f'{data_path}/sample{cement_sample_index}/bse/{bse_zoom_times}/{bse_file_name}.bmp'
        refered_img = cv2.imread(bse_file_path, cv2.IMREAD_GRAYSCALE)
        r_img_height, r_img_width = refered_img.shape
        print(f"r_width: {r_img_width}, r_height: {r_img_height}")

    def get_referred_img_shape(self):
        # (height, width)(rows, column)
        return self.ref_img.shape

    # 加载图像
    def load_img(self):
        self._load_itk()
        self._load_ref_img()

    # 这个值越大越好 空间相关性
    def spatial_correlation(self, img1, img2):
        threshold = self.config.threshold

        img1_threshold = threshold[0]
        img2_threshold = threshold[1]
        bound = bound

        shape = img1.shape

        img1_cp = np.array(img1).astype(float)
        img2_cp = np.array(img2).astype(float)

        # 两张图片分别填充bound之外的值，从而容易筛选出我们想要的元素的个数
        img1_cp[img1 < img1_threshold] = 0
        img2_cp[img2 < img2_threshold] = 255

        lower_bound = bound[0]
        upper_bound = bound[1]

        diff_imgs = np.abs(img1_cp - img2_cp)
        count = np.sum((diff_imgs >= lower_bound) & (diff_imgs <= upper_bound))
        return count/(shape[0] * shape[1])

    def mutual_information(self, image1, image2):
        bins = self.config.bins

        # Calculate the histogram of the images
        hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)

        # Calculate the joint probability distribution
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1) # Marginal for x over y
        py = np.sum(pxy, axis=0) # Marginal for y over x

        # Calculate the mutual information
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0 # Non-zero joint probabilities
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

        return mi

    # 相似度计算
    def similarity(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        lamda_mis = self.config.lamda_mis

        rotation_center = (rotation_center_xy[0], rotation_center_xy[1], x[2].item())
        rotation = (x[3].item(), x[4].item(), x[5].item())
        slice_indeces = (int(x[0]), int(x[1]), int(x[2]))
        slice_img = Tools.get_slice_from_itk(rotation_center, rotation, slice_indeces)

        mi_value = self.mutual_information(slice_img, self.refered_img)
        spation_info = self.spatial_correlation(slice_img, self.refered_img)
        mis = mi_value ##+ lamda_mis * spation_info
        return mis, slice_img


    def registrate(self):
        value, result = self.pso_framework()