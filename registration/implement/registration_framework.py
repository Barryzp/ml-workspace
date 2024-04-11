import numpy as np
import torch
import cv2
import itk
from functools import partial
from utils.tools import Tools


class Registration:
    def __init__(self, config) -> None:
        self.itk_img = None
        self.refered_img = None
        self.moving_image = None
        self.masked_img = None
        self.optim_framework = None
        self.config = config
        self.load_img()

    def set_optim_algorithm(self, optim):
        self.optim_framework = optim
        height, width = self.get_referred_img_shape()
        # 需要绑定实例对象
        similarity_fun = partial(self.similarity)
        optim.set_init_params((width, height), similarity_fun)

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
        self.refered_img = cv2.imread(bse_file_path, cv2.IMREAD_GRAYSCALE)
        r_img_height, r_img_width = self.refered_img.shape
        print(f"r_width: {r_img_width}, r_height: {r_img_height}")

    def get_referred_img_shape(self):
        # (height, width)(rows, column)
        return self.refered_img.shape

    # 加载遮罩图像
    def load_masked_img(self):
        data_path = self.config.data_path
        bse_zoom_times = self.config.bse_zoom_times
        cement_sample_index = self.config.cement_sample_index
        if self.config.masked:
            
            masked_path = f"{data_path}/sample{cement_sample_index}/bse/{bse_zoom_times}/"
            pass
        pass

    # 加载图像
    def load_img(self):
        self._load_ref_img()
        if self.config.mode == "2d":
            self._load_moving_img()
        else:
            self._load_itk()

    # 这个值越大越好 空间相关性
    def spatial_correlation(self, img1, img2):
        threshold = self.config.threshold
        bound = self.config.bound

        img1_threshold = threshold[0]
        img2_threshold = threshold[1]

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

    def similarity_only_mask_spatial(self, x):
        pass

    def similarity_2d(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        lamda_mis = self.config.lamda_mis

        image = self.moving_image
        r_height, r_width = self.get_referred_img_shape()
        f_height, f_width = self.get_moving_img_shape()

        # 步骤 2: 旋转图像
        # 设置旋转中心为图像中心，旋转45度，缩放因子为1
        angle = x[2].item()
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center_xy, angle, scale)     
        # 应用旋转
        rotated_image = cv2.warpAffine(image, rotation_matrix, (f_width, f_height))     

        # 步骤 4: 裁剪图像
        # 设置裁剪区域
        x, y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = rotated_image[y:y+h, x:x+w]
        mi_value = self.mutual_information(cropped_image, self.refered_img)

        spation_info = self.spatial_correlation(cropped_image, self.refered_img)
        mis = mi_value + lamda_mis * spation_info
        return mis, cropped_image


    def similarity_3d(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        lamda_mis = self.config.lamda_mis

        height, width = self.get_referred_img_shape()

        rotation_center = (rotation_center_xy[0], rotation_center_xy[1], x[2].item())
        rotation = (x[3].item(), x[4].item(), x[5].item())
        slice_indeces = (int(x[0]), int(x[1]), int(x[2]))
        slice_img = Tools.get_slice_from_itk(self.itk_img, rotation_center, rotation, slice_indeces, (width, height))

        mi_value = self.mutual_information(slice_img, self.refered_img)
        spation_info = self.spatial_correlation(slice_img, self.refered_img)
        mis = mi_value + lamda_mis * spation_info
        return mis, slice_img.reshape((width, height))

    # 相似度计算 HACK 分2d还是3d
    def similarity(self, x):
        if self.config.mode == "2d":
            return self.similarity_2d(x)
        elif self.config.mode == "3d":
            return self.similarity_3d(x)

    def registrate(self):
        return self.optim_framework.run()