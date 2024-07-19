import cv2, os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from utils.tools import Tools
from utils.segmentation_kms import SegmentationKMS
from utils.common_config import CommonConfig
from utils.visualize import VisualizeData
from dtos.patch_info import PatchCfg, PyramidCfg

class ImageProcess:
    def __init__(self, config) -> None:
        self.config = config
        sample_id = self.config.cement_sample_index
        zoom_times = self.config.bse_zoom_times
        times = zoom_times // 100
        suffix = self.config.zoom_bse_index

        self.file_name_pref = f"{sample_id}-{times}-{suffix}"
        bse_save_path = f"{self.config.data_save_root}/sample{sample_id}/bse/s{self.config.sample_bse_index}/{zoom_times}"
        self.ct_processed_save_path = f"{self.config.ct_src_root}/sample{sample_id}/ct/matched"
        self.bse_save_path = bse_save_path
        self.bse_src_path = f"{self.config.bse_src_root}/{sample_id}/S{self.config.sample_bse_index}/{zoom_times}/{self.file_name_pref}.bmp"

        self.bse_clahe = cv2.createCLAHE(clipLimit=self.config.bse_clipLimit, 
                                         tileGridSize=(self.config.bse_tileGridSize[0], self.config.bse_tileGridSize[1]))

        self.ct_clahe = cv2.createCLAHE(clipLimit=self.config.ct_clipLimit, 
                                         tileGridSize=(self.config.ct_tileGridSize[0], self.config.ct_tileGridSize[1]))

        self.segmentation = SegmentationKMS()

    def save_cfg(self):
        save_file_name = f"{self.file_name_pref}-config.yaml"
        Tools.save_obj_yaml(self.bse_save_path, save_file_name, self.config)

    # 裁减出BSE的ROI，并且进行下采样
    def crop_circle_with_bar(self, center_offset, rect, crop_radius, bar_height):
        # 加载图像（替换为您的图像路径）
        image_path = self.bse_src_path  # 替换为您的图像路径
        downsamp_save_path = f'{self.bse_save_path}'  # 单单裁剪的图像路径，只是把底下的那个比例尺条裁剪去，以及下采样
        downsamp_save_filename = f'{self.file_name_pref}-downsamp.bmp'

        scale_bar_height = bar_height
        rect_left = rect[0]
        rect_right = rect[1]

        # 打开图像
        image = Image.open(image_path).convert('L')  # 确保图像是灰度的
        width, height = image.size
        center = ((width // 2) + center_offset[0] , (height // 2) + center_offset[1])
        # 将无效的成像去除掉
        result_image = Tools.crop_circle(image, radius = crop_radius, center=center)

        # 再对底部区域进行裁剪
        crop_rect = (rect_left, 0, rect_left + rect_right, result_image.height - scale_bar_height)
        result_image = Tools.crop_rectangle(result_image, crop_rect)

        # 缩放倍数
        sample_times = self.config[f'subsample_{self.config.bse_zoom_times}']
        # 进行下采样
        # 新尺寸
        new_size = (round(result_image.width / sample_times) , round(result_image.height / sample_times))
        # 使用双三次插值下采样图像
        resized_image = result_image.resize(new_size, Image.BICUBIC)
        # 保存图像为BMP格式
        Tools.check_file(downsamp_save_path, downsamp_save_filename)
        resized_image.save(f'{downsamp_save_path}/{downsamp_save_filename}', format='BMP')
        # Image对象
        return resized_image
    
    def crop_roi_rect(self, image, roi_size, offset):
        roi_save_path = f'{self.bse_save_path}/{self.file_name_pref}-downsamp-roi.bmp'

        # 裁剪时偏移量
        roi_cropped_offset = offset

        width_ = image.width
        height_ = image.height
        center_new = (roi_cropped_offset[0] + width_ // 2, 
                      roi_cropped_offset[1] +  height_ // 2)
        rect_roi_width = roi_size[0]
        rect_roi_height = roi_size[1]

        # 裁剪为我们的兴趣域大小
        roi_rect = (center_new[0] - rect_roi_width//2, center_new[1] - rect_roi_height//2,
                    center_new[0] + rect_roi_width//2, center_new[1] + rect_roi_height//2)
        roi_result = Tools.crop_rectangle(image, roi_rect)
        roi_result.save(roi_save_path, format='BMP')
        # Image对象
        return roi_result

    # 增强CT图像, ct_img是np_array形式的
    def enhanced_ct(self, ct_img):
        enhanced_ct = self.ct_clahe.apply(ct_img)
        return enhanced_ct

    def segment_ct(self, ct_img_enhanced, cls_num=4, random = None):
        return self.segmentation.kmeans_image_segmentation(ct_img_enhanced, cls_num, random)
    
    # 增强BSE区域，这是只裁剪了原型区域的，并没有剪切为矩形区域
    def enhanced_unrect_bse(self, show_result = True):
        roi_enhanced_save_path = f'{self.bse_save_path}/{self.file_name_pref}-enhanced-roi.bmp'
        
        center_offset = (self.config.center_offset_x, self.config.center_offset_y)
        rect = (self.config.rect_left, self.config.rect_right)
        bar_height = self.config.scale_bar_height
        crop_radius = self.config.crop_radius
        roi_result = self.crop_circle_with_bar(center_offset, rect, crop_radius, bar_height)
       
        clahe_image = self.bse_clahe.apply(roi_result)

        roi_enhanced = Image.fromarray(clahe_image.astype('uint8'), 'L')  # 'RGB' for color images
        roi_enhanced.save(roi_enhanced_save_path, format="BMP")

        if show_result:
            roi_hist = cv2.calcHist([roi_result], [0], None, [256], [0, 256])
            clahe_hist = cv2.calcHist([clahe_image], [0], None, [256], [0, 256])

            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1), plt.imshow(roi_result, cmap='gray', vmin=0, vmax=255), plt.title('roi_result')
            plt.subplot(2, 2, 3), plt.imshow(clahe_image, cmap='gray', vmin=0, vmax=255), plt.title('clahe_image')
            plt.subplot(2, 2, 2), plt.bar(range(256), roi_hist.ravel(), width=1), plt.xlim([0, 256])
            plt.subplot(2, 2, 4), plt.bar(range(256), clahe_hist.ravel(), width=1), plt.xlim([0, 256])

        return roi_enhanced

    # 增强ROI区域，注意，是裁剪为矩形后的图像
    def crop_enhaned_bse(self, show_result = True):
        roi_enhanced_save_path = f'{self.bse_save_path}/{self.file_name_pref}-enhanced-roi.bmp'
        roi_complete_enhanced_save_path = f'{self.bse_save_path}/{self.file_name_pref}-enhanced-complete-roi.bmp'

        center_offset = (self.config.center_offset_x, self.config.center_offset_y)
        rect = (self.config.rect_left, self.config.rect_right)
        bar_height = self.config.scale_bar_height
        crop_radius = self.config.crop_radius
        downsample_img = self.crop_circle_with_bar(center_offset, rect, crop_radius, bar_height)
       
        roi_result = self.crop_roi_rect(downsample_img, 
                                        [self.config.rect_roi_width, self.config.rect_roi_height],
                                        self.config.roi_cropped_offset)
        
        roi_result = np.array(roi_result)
        downsample_img = np.array(downsample_img)

        clahe_image = self.bse_clahe.apply(roi_result)
        clahe_img_complete = self.bse_clahe.apply(downsample_img)

        roi_enhanced = Image.fromarray(clahe_image.astype('uint8'), 'L')  # 'RGB' for color images
        roi_enhanced.save(roi_enhanced_save_path, format="BMP")

        roi_enhanced_comp = Image.fromarray(clahe_img_complete.astype('uint8'), 'L')  # 'RGB' for color images
        roi_enhanced_comp.save(roi_complete_enhanced_save_path, format="BMP")

        if show_result:
            roi_hist = cv2.calcHist([roi_result], [0], None, [256], [0, 256])
            clahe_hist = cv2.calcHist([clahe_image], [0], None, [256], [0, 256])

            plt.figure(figsize=(12, 12))
            plt.subplot(2, 2, 1), plt.imshow(roi_result, cmap='gray', vmin=0, vmax=255), plt.title('roi_result')
            plt.subplot(2, 2, 3), plt.imshow(clahe_image, cmap='gray', vmin=0, vmax=255), plt.title('clahe_image')
            plt.subplot(2, 2, 2), plt.bar(range(256), roi_hist.ravel(), width=1), plt.xlim([0, 256])
            plt.subplot(2, 2, 4), plt.bar(range(256), clahe_hist.ravel(), width=1), plt.xlim([0, 256])

        return roi_enhanced, roi_enhanced_comp
    
    # image是cv2读取的numpy对象
    def kms_segmentation(self, image, id="", random=None):
        cls_num = self.config.class_num
        save_path = self.bse_save_path
        file_pref = self.file_name_pref

        for i in range(cls_num):
            classified_num = i + 2
            seg_result, bin_img = self.segmentation.kmeans_image_segmentation(image, classified_num, random)
            classified_path = f"{save_path}/{file_pref}-{id}-kms{classified_num}.bmp"
            bin_classified_path = f"{save_path}/{file_pref}-{id}-bin-kms{classified_num}.bmp"
            cv2.imwrite(classified_path, seg_result)
            cv2.imwrite(bin_classified_path, bin_img)

    # bse_img_enhanced, bin_img 都是numpy
    def crop_processed_bse_bin_images(self, bse_img_enhanced, bin_img, left_top, cropped_size, offset, suffix=""):
        save_path = self.bse_save_path
        file_pref = self.file_name_pref
        
        bse_img = Image.fromarray(bse_img_enhanced)
        masked_img = Image.fromarray(bin_img)

        center = (bse_img.width // 2, bse_img.height // 2)
        start_left, start_top = left_top[0], left_top[1]
        cropped_width, cropped_height = cropped_size[0], cropped_size[1]
        offset_x, offset_y = offset[0], offset[1]

        center_left = center[0] + offset_x - cropped_width // 2
        center_top = center[1] + offset_y - cropped_height // 2

        rect = (center_left, center_top, center_left+cropped_width, center_top+cropped_height)

        cropped_bse_img = bse_img.crop(rect)
        cropped_bse_img = cropped_bse_img.convert('L')
        cropped_masked_img = masked_img.crop(rect)
        cropped_masked_img = cropped_masked_img.convert('L')

        cropped_bse_img_np = np.array(cropped_bse_img)
        cropped_masked_img_np = np.array(cropped_masked_img)
        masked_white_region = cropped_bse_img_np & cropped_masked_img_np

        cropped_bse_img.save(f"{save_path}/{file_pref}-matched-bse{suffix}.bmp", format="BMP")
        cropped_masked_img.save(f"{save_path}/{file_pref}-matched-masked{suffix}.bmp", format="BMP")

        return cropped_bse_img, cropped_masked_img, masked_white_region

    def seg_and_crop_masked(self, image, random=None, id = "enhanced-roi", final_res_suffix = ""):
        # 1. 对图像进行分割
        self.kms_segmentation(image, id, random)
        path_pref = f"{self.bse_save_path}/{self.file_name_pref}"
        classfied_img_path = f"{path_pref}-{id}-kms{self.config.mask_classfied_num}.bmp"
        kms = cv2.imread(classfied_img_path, cv2.IMREAD_GRAYSCALE)

        # 2. 计算联通区域筛选出较大的颗粒
        filterred_image, _ = self.segmentation.filter_small_size_out(kms, self.config.particle_quantile)
        cv2.imwrite(f"{path_pref}-{id}-masked.bmp", filterred_image)

        kms_cls = self.config.mask_classfied_num

        # 3. 经过一些腐蚀操作去除掉一些细微的颗粒
        processed_img = self.segmentation.morphy_process_kms_image(filterred_image, self.config.gray_cls, self.config.kernel_size)
        cv2.imwrite(f'{path_pref}-{id}-kms{kms_cls}-filter.bmp', processed_img)

        return self.crop_processed_bse_bin_images(
            image, filterred_image, 
            [self.config.start_left, self.config.start_top],
            [self.config.cropped_width, self.config.cropped_height],
            [self.config.offset_x, self.config.offset_y],
            final_res_suffix)
    
    # 二值化图像, gray_cls代表为白色的灰度值
    def binarized_img(self, image, gray_cls):
        neg_cls = image != gray_cls
        positive_cls = image == gray_cls
        image[neg_cls] = 0
        image[positive_cls] = 255
        return image

    # 不裁剪小区域
    def seg_mask(self, image, kms_cls, random=None, id = "roi"):
        # 1. 对图像进行分割
        self.kms_segmentation(image, id, random)
        path_pref = f"{self.bse_save_path}/{self.file_name_pref}"
        classfied_img_path = f"{path_pref}-{id}-bin-kms{kms_cls}.bmp"
        kms_image = cv2.imread(classfied_img_path, cv2.IMREAD_GRAYSCALE)

        # 对图像进行二值化处理 已经二值化处理过了
        # kms_image = self.binarized_img(kms_image, gray_cls)

        # 2. 计算联通区域筛选出较大的颗粒
        filterred_image, diameter_interval = self.segmentation.filter_small_size_out(kms_image, self.config.particle_quantile)
        cv2.imwrite(f"{path_pref}-{id}-filter.bmp", filterred_image)

        # 3. 经过一些腐蚀扩张操作去除掉一些细微的颗粒
        processed_img = self.segmentation.morphy_process_kms_image(filterred_image, self.config.kernel_size)

        # 4. 腐蚀扩张操作会留下一些离散的孔隙，在把这些个孔隙去掉
        processed_img = Tools.filter_by_diameter_bin_img(processed_img, diameter_interval)
        diameter_interval = Tools.get_diameter_interval(processed_img)
        cv2.imwrite(f'{path_pref}-{id}-mask.bmp', processed_img)

        return processed_img, diameter_interval


    # bse图像匹配前预处理
    def matched_bse_img_processed(self):
        # 裁剪并增强
        roi_enhanced, roi_enhanced_comp = self.crop_enhaned_bse()

        random_state = self.config.kmeans_random_status
        rect_kms_cls = self.config.mask_rect_classfied_num
        comp_kms_cls = self.config.mask_comp_classfied_num
        
        # 分割并保存（使用粒径的方式）
        self.seg_mask(np.array(roi_enhanced), rect_kms_cls, random_state)
        roi_comp_img, diameter_interval = self.seg_mask(np.array(roi_enhanced_comp), comp_kms_cls, random_state, "comp")

        max_area, min_area = diameter_interval[1], diameter_interval[0]
        self.config.size_threshold_bse_min = min_area
        self.config.size_threshold_bse_max = max_area
        self.save_cfg()

    def load_bse_preprocessd_cfg(self):
        bse_save_path = self.bse_save_path
        config_name = f"{self.file_name_pref}-config.yaml"
        bse_config = Tools.load_yaml_config(f"{bse_save_path}/{config_name}")
        return bse_config

    # ct图像匹配前预处理
    def matched_ct_img_processed(self):

        # 读取BSE配置，获取标准BSE图像中的水泥颗粒的大小范围
        bse_config = self.load_bse_preprocessd_cfg()

        max_diameter, min_diameter = bse_config.size_threshold_bse_max, bse_config.size_threshold_bse_min
        # 保留较大的颗粒，让图像可以具有多种用途
        max_diameter = max_diameter * 100
        diameter_ratio = bse_config.size_threshold_ratio[1]
        min_diameter = min_diameter / diameter_ratio

        cement_id = self.config.cement_sample_index
        sample_range = CommonConfig.get_range(cement_id)
        total_image_num = sample_range[1] - sample_range[0]
        sample_interval = self.config.sample_interval
        # 先进行基操，看能够到达什么水平
        loop_times = total_image_num // sample_interval

        # 删除掉前面的几张图像
        init_interval_index = self.config.init_interval_index
        # 删除后面几张图片
        end_interval_index = self.config.end_interval_index
        start_index = sample_range[0] + sample_interval * init_interval_index
        loop_times = loop_times - init_interval_index - end_interval_index

        kmeans_random = self.config.kmeans_random_status
        ct_seg_cls = self.config.mask_comp_classfied_num

        temp_mask_img = None
        # 记录所有的编号
        for i in range(loop_times):
            slice_index = start_index + i * sample_interval
            ori_ct_img = CommonConfig.get_cement_ct_slice(cement_id, slice_index)
            
            save_enhanced_img_name = f"{slice_index}_enhanced_ct.bmp"
            save_bin_img_name = f"{slice_index}_mask_ct.bmp"
            save_test_seg_img_name = f"{slice_index}_segment_ct.bmp"
            save_test_temp_img_name = f"{slice_index}_temp_ct.bmp"

            # 对比度增强
            if self.config.enhanced : enhanced_ct = self.enhanced_ct(ori_ct_img)
            enhanced_ct = ori_ct_img
            Tools.save_img(self.ct_processed_save_path, save_enhanced_img_name, enhanced_ct)
            # 分割图像
            cls_ct, bin_ct = self.segment_ct(enhanced_ct, ct_seg_cls, kmeans_random)
            if self.config.save_temp_res : Tools.save_img(self.ct_processed_save_path, save_test_seg_img_name, cls_ct)

            # HACK 过时方法了            
            # if temp_mask_img is not None:
            #     and_img = temp_mask_img & cls_ct
            #     if self.config.save_temp_res : Tools.save_img(self.ct_processed_save_path, save_test_temp_img_name, and_img)
            #     ct_gray_cls,_ = Tools.find_ith_frequent_element(and_img, 2)
            
            # 二值化图像
            ct_bin_img = bin_ct # self.binarized_img(cls_ct, ct_gray_cls)
            # 2. 计算联通区域筛选出较大的颗粒
            filterred_image = Tools.filter_by_diameter_bin_img(ct_bin_img, [min_diameter, max_diameter])
            
            # 3. 经过一些腐蚀操作去除掉一些细微的颗粒
            processed_img = self.segmentation.morphy_process_kms_image(filterred_image, self.config.kernel_size)
            Tools.save_img(self.ct_processed_save_path, save_bin_img_name, processed_img)

            print(f"{slice_index} processed.")
            temp_mask_img = processed_img

    # 保存筛除掉对应大小的水泥颗粒图像
    def save_filterred_diff_size(self, ori_bin_img, slice_id, std_min, std_max):
        ratios = self.config.size_threshold_ratio

        # 寻找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_bin_img, 4, cv2.CV_32S)

        def filter_img(img, min, max):
            # 遍历每个联通区域
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < min or area > max:
                    # 如果面积不在指定范围内，将该区域填充为背景色
                    img[labels == i] = 0

            return img

        for ratio in ratios:
            max_area = std_max * ratio
            min_area = std_min
            save_bin_img_filterred_name = f"{slice_id}_mask_filterred_{ratio}_ct.bmp"
            processed_img = filter_img(np.copy(ori_bin_img), min_area, max_area)
            Tools.save_img(self.ct_processed_save_path, save_bin_img_filterred_name, processed_img)

    # 计算在剪切后的CT图像中的mi值
    def compute_mi_in_cropped(self, ct_img, bse_img, crop_rect, rot):
        ct_size = self.config.ground_ct_size
        cropped_bse_rect_in_ct = Tools.crop_rotate_mi(ct_img, 
                                       [ct_size[0] * 0.5, ct_size[0] * 0.5],
                                       ct_size,
                                       rot,
                                       crop_rect
                                       )
        mi = Tools.mutual_information(cropped_bse_rect_in_ct, bse_img)
        return mi, cropped_bse_rect_in_ct

    # 正向或者反向获取最优的slice
    def get_best_slice_idx(self, start_idx, interval, refer_img, crop_rect, rot, forward = True):
        delta = -1
        if forward : delta = 1

        start_ct_img = Tools.get_ct_img(self.config.cement_sample_index, start_idx)
        best_slice_index = start_idx
        best_mi, _ = self.compute_mi_in_cropped(start_ct_img, refer_img, crop_rect, rot)

        for i in range(interval):
            slice_idx = start_idx + (i + 1)*delta
            ct_img = Tools.get_ct_img(self.config.cement_sample_index, slice_idx)

            mi, _ = self.compute_mi_in_cropped(ct_img, refer_img, crop_rect, rot)
            print(f"slice: {slice_idx}, mi: {mi}")
            if mi > best_mi :
                best_mi = mi
                best_slice_index = slice_idx

        return best_mi, best_slice_index


    # bse roi区域在到原始CT中对应的坐标
    def bse_roi_remapping_in_ct(self):
        pos_in_ct = np.array(self.config.matched_translate) * self.config.downsample_times

        # 匹配的BSE图像的剪切
        matched_bse_rect = [
            pos_in_ct[0].item(),
            pos_in_ct[1].item(),
            self.config.bse_roi_size[0],
            self.config.bse_roi_size[1]
        ]
        return matched_bse_rect

    def cropped_rect_remapping(self):
        # 剪切区域位置到原始CT图像的重映射
        ct_size = self.config.ground_ct_size
        crop_from_origin_delta_x = (ct_size[0] - self.config.origin_ct_size[0]) * 0.5
        crop_from_origin_delta_y = (ct_size[1] - self.config.origin_ct_size[1]) * 0.5
        # 映射到未下采样时的裁剪坐标点
        pos_in_origin_crop = np.array(self.config.matched_translate) * self.config.downsample_times
        # 映射到原始CT图像中的裁剪坐标点
        crop_ground_ct_pos = pos_in_origin_crop + [crop_from_origin_delta_x, crop_from_origin_delta_y]

        # 匹配图像剪切
        matched_cropped_rect = [crop_ground_ct_pos[0], 
                        crop_ground_ct_pos[1],
                        self.config.origin_matched_size[0],
                        self.config.origin_matched_size[1],
                        ]

        # 匹配的BSE图像的剪切
        matched_bse_rect = [
            crop_ground_ct_pos[0] - self.config.bse_cropped_offset[0] + self.config.origin_matched_size[0] * 0.5,
            crop_ground_ct_pos[1] - self.config.bse_cropped_offset[1] + self.config.origin_matched_size[1] * 0.5,
            self.config.bse_roi_size[0],
            self.config.bse_roi_size[1]
        ]

        return matched_cropped_rect, matched_bse_rect

    def get_best_slice_idx_from_ct(self, init_slice_index, bse_cropped_matched_img, cropped_rect):
        latent_slice_area = self.config.latent_slice_area

        ct_size = self.config.ground_ct_size
        ct_img = Tools.get_ct_img(self.config.cement_sample_index, init_slice_index)
        ct_cropped_matched = Tools.crop_rotate_mi(ct_img, 
                                   [ct_size[0] * 0.5, ct_size[0] * 0.5],
                                   ct_size,
                                   self.config.matched_rotation,
                                   cropped_rect
                                   )

        init_mi = Tools.mutual_information(bse_cropped_matched_img, ct_cropped_matched)
        print(f"init best slice: {init_slice_index}, mi: {init_mi}")

        forward_best_mi, forward_best_idx = self.get_best_slice_idx(init_slice_index, 
                                                               latent_slice_area, 
                                                               bse_cropped_matched_img, 
                                                               cropped_rect, 
                                                               self.config.matched_rotation)
        backward_best_mi, backward_best_idx = self.get_best_slice_idx(init_slice_index, 
                                                               latent_slice_area, 
                                                               bse_cropped_matched_img, 
                                                               cropped_rect, 
                                                               self.config.matched_rotation,
                                                               False)

        cropped_mis = np.array([init_mi, forward_best_mi, backward_best_mi])
        cropped_index = np.array([init_slice_index, forward_best_idx, backward_best_idx])

        best_cropped_array_index = np.argmax(cropped_mis)
        best_cropped_index = cropped_index[best_cropped_array_index]
        best_cropped_mi = cropped_mis[best_cropped_array_index]
        return best_cropped_index, best_cropped_mi
    
    def process_and_save_ct(self, slice_index, interval, file_path):

        ct_size = self.config.ground_ct_size
        angle = self.config.matched_rotation
        center = [ct_size[0] * 0.5, ct_size[0] * 0.5]

        for i in range(interval):
            index = slice_index + i
            ct_img = Tools.get_ct_img(self.config.cement_sample_index, index)
            file_name = f"{index}_enhanced.bmp"
            # 进行旋转
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)     
            rotated_image = cv2.warpAffine(ct_img, rotation_matrix, (ct_size[0], ct_size[1]))
            # 进行对比度增强
            enhanced_img = self.ct_clahe.apply(rotated_image)
            # 进行保存
            Tools.save_img(file_path, file_name, enhanced_img)

    # 选择最佳潜在区域，并保存起来，这是针对于是小的切片的情况下的，也就是从CT中截取了1024*1024大小的，BSE上截取的ROI
    def choose_best_slices_save_tiny_region(self):
        # (1) 剪切区域重映射
        # (2) 获取匹配区域最优的slice
        # (3) 获取整个BSE ROI最优的slice
        # (4) 对最优的BSE进行选择
        # (5) 得到最优区域之后将图像复制到指定文件夹中
        matched_rect_init, matched_rect_bse = self.cropped_rect_remapping()
        matched_bse_refer_path, matched_bse_refer_filename = Tools.get_processed_bse_path(self.config)
        file_pref = f"{matched_bse_refer_path}/{matched_bse_refer_filename}"
        matched_bse_img = cv2.imread(f"{file_pref}-enhanced-roi.bmp", cv2.IMREAD_GRAYSCALE)
        matched_bse_cropped_img = cv2.imread(f"{file_pref}-matched-bse.bmp", cv2.IMREAD_GRAYSCALE)
        
        middle_matched_slice = self.config.matched_slice_index
        proper_matched_slice_index, proper_matched_mi = self.get_best_slice_idx_from_ct(
            middle_matched_slice, matched_bse_cropped_img, matched_rect_init
        )
        print(f"best cropped index: {proper_matched_slice_index}, best cropped mi: {proper_matched_mi}")

        best_matched_slice_index, best_matched_mi = self.get_best_slice_idx_from_ct(
            proper_matched_slice_index, matched_bse_img, matched_rect_bse
        )
        print(f"best cropped index: {best_matched_slice_index}, best cropped mi: {best_matched_mi}")
        
        file_path = f"{self.config.data_save_root}/sample{self.config.cement_sample_index}/ct/s{self.config.sample_bse_index}"
        
        interval = self.config.latent_slice_area * 2
        start_id = best_matched_slice_index - self.config.latent_slice_area
        self.process_and_save_ct(start_id, interval, file_path)

        self.config.best_bse_slice = best_matched_slice_index.item()
        self.config.best_start_id = start_id.item()
        self.config.best_end_id = (start_id + interval - 1).item()
        cfg_name = f"cement_{self.config.cement_sample_index}_s{self.config.sample_bse_index}.yaml"
        Tools.save_obj_yaml(file_path, cfg_name, self.config)

        return best_matched_slice_index

    # 转移最佳切片所在最优区域
    def choose_best_volume_for_fine_reg(self):
        # 咱们就扩大范围，以对应的切片为中心上下总共80层作为结果复制过去，80层的原因是因为下采样了8倍，这样好复制一些
        middle_slice_index = self.config.matched_slice_index
        latent_volume_depth = self.config.latent_slice_area
        half_depth = latent_volume_depth // 2
        start_slice_index = middle_slice_index - half_depth
        end_slice_index = middle_slice_index + half_depth - 1
        volume_slice_interval = [start_slice_index, end_slice_index]
        self.config.volume_slice_interval = volume_slice_interval
        
        # 保存一下配置
        file_path = f"{self.config.data_save_root}/sample{self.config.cement_sample_index}/ct/s{self.config.sample_bse_index}"
        cfg_name = f"cement_{self.config.cement_sample_index}_s{self.config.sample_bse_index}.yaml"
        Tools.save_obj_yaml(file_path, cfg_name, self.config)

        # 复制图片过去，不进行变换操作，因为参数咱们还要接着用
        for i in range(latent_volume_depth):
            index = start_slice_index + i
            ct_img = Tools.get_ct_img(self.config.cement_sample_index, index)
            enhanced_file_name = f"{index}_enhanced.bmp"
            ori_file_name = f"slice_{index}.bmp"
            Tools.save_img(file_path, ori_file_name, ct_img)
            enhanced_img = self.ct_clahe.apply(ct_img)
            # 进行保存
            Tools.save_img(file_path, enhanced_file_name, enhanced_img)


    # HACK 过时的方法，逐层逐片搜索 使用截取的BSE ROI图像来选择最佳区域(此时是这样的：bse_roi在整个ct区域上进行)
    def choose_best_slices_bse_roi(self):
        # 同样的逻辑
        # (1) 剪切区域重映射
        # (2) 获取匹配区域最优的slice
        # (3) 获取整个BSE ROI最优的slice
        # (4) 对最优的BSE进行选择
        # (5) 我在想有一步确定最佳Z切片的需要放在这一步吗？并且其最佳切片的值还要进一步保存到文件里
        #       还有一个问题：就是固定Z可以进一步减少解空间的大小。
        # (6) 得到最优区域之后将图像进行剪切（减小内存大小）并复制到指定文件夹中
        matched_rect_bse = self.bse_roi_remapping_in_ct()
        matched_bse_refer_path, matched_bse_refer_filename = Tools.get_processed_bse_path(self.config)
        file_pref = f"{matched_bse_refer_path}/{matched_bse_refer_filename}"

        matched_bse_img = cv2.imread(f"{file_pref}-enhanced-roi.bmp", cv2.IMREAD_GRAYSCALE)
        middle_matched_slice = self.config.matched_slice_index
        proper_matched_slice_index, proper_matched_mi = self.get_best_slice_idx_from_ct(
            middle_matched_slice, matched_bse_img, matched_rect_bse
        )
        print(f"best cropped index: {proper_matched_slice_index}, best cropped mi: {proper_matched_mi}")

        # HACK 来测试一波
        patches_info = self.sapwn_patches(proper_matched_slice_index, matched_bse_img)

        # HACK 循环迭代寻找各个Patch所处在的最优切片
        patches_mi = self.set_patch_best_slice_idx(proper_matched_slice_index, patches_info)

        # self.visualize_patch_info(patches_info)
        return proper_matched_slice_index, patches_mi
    

    # HACK 后期再改良，目前的版本只有原倍数下的图像
    def compute_patch_mi(self, patch_info, ct_img):
        patch_in_times1 = patch_info.pyramids[0]
        bse_img_size = patch_in_times1.patch_size

        r_img = patch_in_times1.r_img
        rect = [patch_in_times1.delta_translate_ct[0], patch_in_times1.delta_translate_ct[1],
                bse_img_size[0], bse_img_size[1]]
        rotation = self.config.matched_rotation
        mi, cropped_img = self.compute_mi_in_cropped(ct_img, r_img, rect, rotation)
        return mi, cropped_img

    def set_patch_best_slice_idx(self, middle_best_slice_index, patches_info):
        latent_slice_area = self.config.latent_slice_area
        ct_size = self.config.ground_ct_size
        # how todo:
        # (1) 从底层到上层，index由低到高，比如说：目前最佳是580，那么id从580-latent_slice_area ~ 580+latent_slice_area
        # (2) 遍历patches_info，计算每个patch的mi
        # (3) 将每个patch对应切片CT裁剪区域的mi保存起来，用id作为索引，格式如：
        # (4) 将每个层的mi都保存一下
        patches_mi = []
        for patch_info in patches_info:
            patch_mi = {
                "patch_id" : patch_info.id,
                "patch_info" : patch_info,
                "mis" : []
            }
            patches_mi.append(patch_mi)

        start_index_ct = middle_best_slice_index - latent_slice_area
        total_area = latent_slice_area * 2

        for i in range(total_area):
            ct_slice = start_index_ct + i
            ct_img = Tools.get_ct_img(self.config.cement_sample_index, ct_slice)
    
            for patch_mi in patches_mi:
                patch_info = patch_mi.get("patch_info")
                mi, cropped_img = self.compute_patch_mi(patch_info, ct_img)
                slice_mi_item = {
                    "slice_index":ct_slice,
                    "mi" : mi,
                    "cropped_img" : cropped_img
                }
                patch_mi.get("mis").append(slice_mi_item)

        return patches_mi


    def visualize_patch_info(self, patches_info, downsamples_times=1, bse_or_ct = "bse"):
        imgs = []
        labels = []
        count = 1
        for patch_info in patches_info:
            img = patch_info.pyramids[0].r_img
            imgs.append(img)
            labels.append(f"{count}")
            count+=1
        VisualizeData.show_concate_imgs(imgs, 5)

    # 从BSE图像中截取某个尺寸大小的区域
    def crop_from_bse(self, bse_img, rect):
        pos_x, pos_y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])  # 裁剪位置和大小
        cropped_image = bse_img[pos_y:pos_y+h, pos_x:pos_x+w]
        return cropped_image

    def sapwn_patches(self, best_slice, bse_img):
        # 根据设定的patch大小来对图像进行分patch处理（部分patch可能会大于也会小于标准的patch大小）
        patch_size = self.config.patch_size
        half_patch_size = patch_size * 0.5
        bse_size = self.config.bse_roi_size # width, height
        bse_pos_in_ct = self.config.matched_translate # bse图像左上角位于ct图像的位置，以左上角为原点

        width_remain = bse_size[0] % patch_size
        height_remain = bse_size[1] % patch_size

        # 用来判断宽或者高的多余的patch是否能够作为单独的一格，大于设定的patch大小的一半就行（这是一个先验）
        does_width_single = False
        does_height_single = False

        if width_remain > half_patch_size: does_width_single = True
        if height_remain > half_patch_size: does_height_single = True

        cols = bse_size[0] // patch_size
        rows = bse_size[1] // patch_size
        if does_width_single : cols += 1
        if does_height_single : rows += 1

        start_x = bse_pos_in_ct[0]
        start_y = bse_pos_in_ct[1]
        x_delta = 0
        y_delta = 0

        # patch 的大小
        patch_width = patch_size
        patch_height = patch_size

        start_id = 1
        patches_info = []

        for i in range(rows):
            y_delta = patch_size * i

            if i == rows - 1:
                if does_height_single :
                    patch_height = height_remain
                else:
                    patch_height += height_remain

            patch_width = patch_size
            for j in range(cols):
                x_delta = patch_size * j

                if j == cols - 1:
                    if does_width_single :
                        patch_width = width_remain
                    else:
                        patch_width += width_remain

                translate_ct = [start_x+x_delta, start_y+y_delta]
                translate_bse = [x_delta, y_delta]
                patch_info = self.spawn_preprocess_patch_info(start_id, best_slice, bse_img, 
                                                              translate_bse, translate_ct,
                                                              [patch_width, patch_height])
                patches_info.append(patch_info)
                start_id += 1
                # print(f"position: {x, y}")
                # print(f"patch size: {patch_width, patch_height}")

        return patches_info

    # 生成预处理的patch信息，这里的patch信息是未经过下采样的图像，也就是downsample_times为1时的配置信息
    def spawn_preprocess_patch_info(self, id, best_slice, bse_img, translate_bse, translate_ct, size):
        patch_info = PatchCfg()

        pyramids = []
        for i in range(3):
            pyramid = PyramidCfg.buildPyramid(2**i, None, None, None, None)
            pyramids.append(pyramid)

        pyramid1 = pyramids[0]
        crop_rect = [translate_bse[0], translate_bse[1], size[0], size[1]]
        pyramid1.r_img = self.crop_from_bse(bse_img, crop_rect)
        pyramid1.patch_size = size
        pyramid1.delta_translate_bse = translate_bse
        pyramid1.delta_translate_ct = translate_ct

        patch_info.set_info(id, best_slice, pyramids)
        return patch_info

    def search_latent_area(self):
        best_slice = self.choose_best_slices_bse_roi()
        patchs_info = self.sapwn_patches(best_slice, )
