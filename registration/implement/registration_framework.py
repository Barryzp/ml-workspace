import numpy as np
import torch
import cv2
import itk
from functools import partial
from utils.tools import Tools


class Registration:
    def __init__(self, config, ct_index_array = None) -> None:
        self.itk_img = None
        self.refered_img = None
        self.moving_image = None
        self.masked_img = None

        # region 未经过下采样的原始图像
        # 这两个是bse图像中的
        self.ref_img_ori = None
        self.ref_mask_ori = None
        # 这两个是ct图像中的，在匹配的时候是为空的
        self.mov_img_ori = None
        self.msk_img_ori = None
        # endregion

        self.optim_framework = None
        self.config = config

        if self.config.mode == "matched":
            # 匹配过程中ct的索引数组
            self.ct_index_array = ct_index_array
            self.matched_moving_imgs = {}

        self.load_img()
        self.set_config_delta()
    
    def set_config_delta(self):
        mode = self.config.mode
        if mode == "matched" or mode == "2d":
            bse_height, bse_width = self.get_referred_img_shape()
            ct_height, ct_width = self.config.cropped_ct_size[0], self.config.cropped_ct_size[1]
            self.config.rotation_center_xy = [ct_width/2, ct_height/2]
            translate_x = ct_width - bse_width
            translate_y = ct_height - bse_height
            self.config.translate_delta[0] = translate_x
            self.config.translate_delta[1] = translate_y


    def set_optim_algorithm(self, optim, ct_matching_slice_index = None):
        self.optim_framework = optim

        height, width = self.get_referred_img_shape()
        # 需要绑定实例对象
        similarity_fun = partial(self.similarity)
        optim.set_init_params((width, height), similarity_fun, ct_matching_slice_index)

    def _load_matched_moving_img(self):
        ct_index_array = self.ct_index_array
        data_path = self.config.data_path
        cement_sample_index = self.config.cement_sample_index
        ct_image_path = f"{data_path}/sample{cement_sample_index}/ct/matched"

        for index in ct_index_array:
            file_path = f"{ct_image_path}/{index}_mask_ct.bmp"
            moving_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.config.downsampled: moving_img = Tools.downsample_image(moving_img, self.config.downsample_times)
            self.matched_moving_imgs[index] = moving_img
            if self.config.mode == "matched": self.config.cropped_ct_size = moving_img.shape
            

    def _load_moving_img(self):
        if self.config.debug:
            self.moving_image = cv2.imread(self.config.debug_ct_path, cv2.IMREAD_GRAYSCALE)
            self.msk_img_ori = np.copy(self.moving_image)
            self.mov_img_ori = cv2.imread(self.config.debug_ct_ori_path, cv2.IMREAD_GRAYSCALE)
            if self.config.downsampled: self.moving_image = Tools.downsample_image(self.moving_image, self.config.downsample_times)
            self.config.cropped_ct_size = self.moving_image.shape
            return

        data_path = self.config.data_path
        cement_sample_index = self.config.cement_sample_index
        sample_bse_index = self.config.sample_bse_index
        ct_2d_index = self.config.ct_2d_index

        ct_image_path = f"{data_path}/sample{cement_sample_index}/ct/s{sample_bse_index}/enhanced"
        self.moving_image = cv2.imread(f"{ct_image_path}/slice_enhanced_{ct_2d_index}.bmp", cv2.IMREAD_GRAYSCALE)
        self.mov_img_ori = np.copy(self.moving_image)
        if self.config.downsampled: self.moving_image = Tools.downsample_image(self.moving_image, self.config.downsample_times)
        self.config.cropped_ct_size = self.moving_img.shape

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
        if self.config.debug:
            self.refered_img = cv2.imread(self.config.debug_bse_path, cv2.IMREAD_GRAYSCALE)
            self.ref_img_ori = cv2.imread(self.config.debug_bse_ori_path, cv2.IMREAD_GRAYSCALE)
            self.ref_mask_ori = np.copy(self.refered_img)
            if self.config.downsampled: 
                self.refered_img = Tools.downsample_image(self.refered_img, self.config.downsample_times)
                num_labels_m, labels_m, self.stats_r, centroids_m = cv2.connectedComponentsWithStats(self.refered_img, 4, cv2.CV_32S)
            return

        src_path, file_name = Tools.get_processed_referred_path(self.config)
        prefix = file_name
        suffix = self.config.bse_suffix

        bse_file_name = f"{prefix}-{suffix}"
        bse_file_path = f'{src_path}/{bse_file_name}.bmp'
        self.refered_img = cv2.imread(bse_file_path, cv2.IMREAD_GRAYSCALE)
        self.ref_img_ori = np.copy(self.refered_img)

        # HACK 暂时先这么处理吧
        if self.config.mode == "matched":
            bse_mask_path = f"{src_path}/{prefix}-{self.config.mask_suffix}.bmp"
            self.refered_img = cv2.imread(bse_mask_path, cv2.IMREAD_GRAYSCALE)

        if self.config.downsampled: self.refered_img = Tools.downsample_image(self.refered_img, self.config.downsample_times)
        r_img_height, r_img_width = self.refered_img.shape
        print(f"r_width: {r_img_width}, r_height: {r_img_height}")

        if self.config.filter_big_particle:
            num_labels_m, labels_m, self.stats_r, centroids_m = cv2.connectedComponentsWithStats(self.refered_img, 4, cv2.CV_32S)

    def get_referred_img_shape(self):
        # (height, width)(rows, column)
        return self.refered_img.shape

    # 加载遮罩图像
    def _load_masked_img(self):
        if self.config.debug and self.config.masked:
            self.masked_img = cv2.imread(self.config.debug_mask_path, cv2.IMREAD_GRAYSCALE)
            if self.config.downsampled: self.masked_img = Tools.downsample_image(self.masked_img, self.config.downsample_times)
            return

        src_path, prefix = Tools.get_processed_referred_path(self.config)

        if self.config.masked:
            file_name = f"{prefix}-{self.config.mask_suffix}.bmp"
            masked_path = f"{src_path}/{file_name}"
            self.masked_img = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)
            self.ref_mask_ori = np.copy(self.masked_img)
            if self.config.downsampled: self.masked_img = Tools.downsample_image(self.masked_img, self.config.downsample_times)

    # 加载图像
    def load_img(self):
        self._load_ref_img()
        print(f"H_Refer: {Tools.caculate_entropy(self.refered_img)}")

        if self.config.masked:
            self._load_masked_img()
        if self.config.mode == "2d":
            self._load_moving_img()
        elif self.config.mode == "3d":
            self._load_itk()
        elif self.config.mode == "matched":
            self._load_matched_moving_img()

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

    # 带遮罩的空间信息
    def spatial_correlation_with_mask(self, img1, img2):
        if self.config.masked == False:
            return 0

        bound = self.config.bound
        lower_bound = bound[0]
        upper_bound = bound[1]

        img1_after_masked = img1 & self.masked_img
        img2_after_masked = img2 & self.masked_img

        mask_white_bool = self.masked_img == 255
        mask_num = mask_white_bool.sum()

        img1_after_masked = np.int16(img1_after_masked)
        img2_after_masked = np.int16(img2_after_masked)

        diff_imgs = np.abs(img1_after_masked - img2_after_masked)
        diff_imgs = np.uint8(diff_imgs)
        above_lower_bound = diff_imgs > lower_bound
        less_upper_bound = diff_imgs <= upper_bound
        above_upper_bound = diff_imgs > upper_bound

        count = np.sum(above_lower_bound & less_upper_bound)
        penalty = np.sum(above_upper_bound)

        # 这个下界不能包含了，不然就有问题了，因为上述位运算会出来许多的0
        count = np.sum((diff_imgs > lower_bound) & (diff_imgs <= upper_bound))
        return (count - penalty).item() / mask_num

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

    # 只利用遮罩的空间信息
    def similarity_only_mask_spatial(self, x, ct_matching_slice_index):
        rotation_center_xy = self.config.rotation_center_xy

        image = self.matched_moving_imgs[ct_matching_slice_index]
        r_height, r_width = self.get_referred_img_shape()
        f_height = self.config.cropped_ct_size[0]
        f_width = self.config.cropped_ct_size[1]

        angle = x[2].item()
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center_xy, angle, scale)     
        rotated_image = cv2.warpAffine(image, rotation_matrix, (f_width, f_height))     

        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]
        spatial_info = self.spatial_correlation_with_mask(cropped_image, self.refered_img)
        return spatial_info, cropped_image

    # 只利用遮罩的空间信息
    def similarity_matched_mi(self, x, ct_matching_slice_index, sp_lambda):
        rotation_center_xy = self.config.rotation_center_xy

        image = self.matched_moving_imgs[ct_matching_slice_index]
        r_height, r_width = self.get_referred_img_shape()
        f_height = self.config.cropped_ct_size[0]
        f_width = self.config.cropped_ct_size[1]

        angle = x[2].item()
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center_xy, angle, scale)     
        rotated_image = cv2.warpAffine(image, rotation_matrix, (f_width, f_height))     

        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]
        
        mi = self.mutual_information(cropped_image, self.refered_img)

        sp = self.spatial_correlation_with_mask(cropped_image, self.refered_img)
        weightd_sp = self.config.lamda_mis * sp * sp_lambda
        similar = mi + weightd_sp
        # print(f"sp: {sp}, mi: {mi}, similar: {similar}")
        return similar, cropped_image, mi, sp, weightd_sp 

    def similarity_matched_dice(self, x, ct_matching_slice_index):
        rotation_center_xy = self.config.rotation_center_xy

        image = self.matched_moving_imgs[ct_matching_slice_index]
        r_height, r_width = self.get_referred_img_shape()
        f_height = self.config.cropped_ct_size[0]
        f_width = self.config.cropped_ct_size[1]

        angle = x[2].item()
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center_xy, angle, scale)     
        rotated_image = cv2.warpAffine(image, rotation_matrix, (f_width, f_height))     

        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]
        
        dice = Tools.dice_coefficient(cropped_image, self.refered_img)
        if self.config.filter_big_particle : 
            penalty = Tools.big_particle_penalty(self.stats_r, cropped_image)
            dice = dice - self.config.big_particle_lambda * penalty
        return dice, cropped_image, 0, 0, 0 

    def similarity_jaccard(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        image = self.moving_image
        r_height, r_width = self.get_referred_img_shape()
        f_height, f_width = self.get_moving_img_shape()

        # 步骤 2: 旋转图像、裁剪图像
        angle = x[2].item()

        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = Tools.crop_rotate_mi(image, 
                                          rotation_center_xy, 
                                          (f_width, f_height), 
                                          angle, 
                                          [pos_x, pos_y, w, h])

        # 特殊处理得到的二值化图像，计算jaccard系数
        jaccard_value = Tools.jaccard_index(cropped_image, self.refered_img)
        return jaccard_value, cropped_image, 0, 0, 0

    # 用于匹配的dice系数比较相似度
    def similarity_dice(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        image = self.moving_image
        r_height, r_width = self.get_referred_img_shape()
        f_height, f_width = self.get_moving_img_shape()

        # 步骤 2: 旋转图像、裁剪图像
        angle = x[2].item()

        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = Tools.crop_rotate_mi(image, 
                                             rotation_center_xy, 
                                          (f_width, f_height), 
                                          angle, 
                                          [pos_x, pos_y, w, h])
        # rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]

        # 特殊处理得到的二值化图像，计算其DICE分数，DICE分数其实包含了一定程度上的空间信息
        dice_value = Tools.dice_coefficient(cropped_image, self.refered_img)
        if self.config.filter_big_particle : 
            penalty = Tools.big_particle_penalty(self.stats_r, cropped_image)
            dice_value = dice_value - self.config.big_particle_lambda * penalty

        return dice_value, cropped_image, 0, 0, 0

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
        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]
        mi_value = self.mutual_information(cropped_image, self.refered_img)

        spation_info = self.spatial_correlation_with_mask(cropped_image, self.refered_img)
        weighted_sp = lamda_mis * spation_info
        mis = mi_value + weighted_sp
        return mis, cropped_image, mi_value, spation_info, weighted_sp

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

    def similarity(self, x, ct_matching_slice_index=None, sp_lambda=0):
        if self.config.debug :
            # 反射机制：通过函数名字符串调用对象的成员方法
            return getattr(self, self.config.debug_simiarity)(x)
            # if self.config.debug_simiarity == "similarity_dice":
            #     return self.similarity_dice(x)

        if self.config.mode == "2d":
            return self.similarity_2d(x)
        elif self.config.mode == "3d":
            return self.similarity_3d(x)
        elif self.config.mode == "matched":
            return self.similarity_matched_dice(x, ct_matching_slice_index)

    def save_matched_result(self, position):
        # 对于重采样的重新处理，旋转角度不需要操作的
        crop_x, crop_y = position[0].item(), position[1].item()
        rot = position[-1].item()

        downsample_times = self.config.downsample_times
        height, width = self.get_referred_img_shape()
        rect = np.array([crop_x, crop_y, width, height]) * downsample_times
        # 1. 获取在原始大小遮罩CT图像的mask结果
        result_mask_ct_matched = Tools.rotate_and_crop_img(self.msk_img_ori, rot, rect)
        # 2. 获取在原始大小下CT图像的结果
        result_ct_matched = Tools.rotate_and_crop_img(self.mov_img_ori, rot, rect)
        
        file_path = Tools.get_save_path(self.config)
        mask_file_name = f"1Aa-mask_ct.bmp"
        bse_mask_name = f"1Aa-mask_bse.bmp"
        ct_file_name = f"1Aa-ori_ct.bmp"
        bse_file_name = f"1Aa-ori_bse.bmp"

        Tools.save_img(file_path, ct_file_name, result_ct_matched)
        Tools.save_img(file_path, mask_file_name, result_mask_ct_matched)
        Tools.save_img(file_path, bse_file_name, self.ref_img_ori)
        Tools.save_img(file_path, bse_mask_name, self.ref_mask_ori)

    def registrate(self):
        fitness, best_reg, best_position = self.optim_framework.run()

        if self.config.mode == "2d" and self.config.debug:
            self.save_matched_result(best_position)
        return fitness, best_reg, best_position