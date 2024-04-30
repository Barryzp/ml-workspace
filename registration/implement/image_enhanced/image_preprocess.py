import cv2, os
import numpy as np
import itk, yaml
import pandas as pd
import matplotlib.pyplot as plt

from math import radians
from munch import Munch
from pathlib import Path
from PIL import Image, ImageDraw
from utils.tools import Tools
from utils.segmentation_kms import SegmentationKMS

class ImageProcess:
    def __init__(self, config) -> None:
        self.config = config
        sample_id = self.config.cement_sample_index
        zoom_times = self.config.bse_zoom_times
        times = zoom_times // 100
        suffix = self.config.zoom_bse_index

        self.file_name_pref = f"{sample_id}-{times}-{suffix}"
        bse_save_path = f"{self.config.data_save_root}/sample{sample_id}/bse/s{self.config.sample_bse_index}/{zoom_times}"
        self.bse_save_path = bse_save_path
        self.bse_src_path = f"{self.config.bse_src_root}/{sample_id}/S{self.config.sample_bse_index}/{times}/{self.file_name_pref}.bmp"

        self.segmentation = SegmentationKMS()


    def crop_circle_with_bar(self, center_offset, rect, crop_radius, bar_height):
        # 加载图像（替换为您的图像路径）
        image_path = self.bse_src_path  # 替换为您的图像路径
        downsamp_save_path = f'{self.bse_save_path}/{self.file_name_pref}-downsamp.bmp'  # 单单裁剪的图像路径，只是把底下的那个比例尺条裁剪去，以及下采样

        scale_bar_height = bar_height
        rect_left = rect[0]
        rect_right = rect[1]

        # 打开图像
        image = Image.open(image_path).convert('L')  # 确保图像是灰度的
        width, height = image.size
        center = ((width // 2) + center_offset[0] , (height // 2) + center_offset[1])
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
        resized_image.save(downsamp_save_path, format='BMP')
        # Image对象
        return resized_image
    
    def crop_roi_rect(self, image, roi_size, offset):
        roi_save_path = f'{self.bse_save_path}/{self.file_name_pref}-downsamp-roi.bmp'

        # 裁剪时偏移量
        roi_cropped_offset = offset

        width_ = image.width
        height_ = image.height
        center_new = (roi_cropped_offset[0] + image.width // 2, 
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

    def crop_enhaned_bse(self, show_result = True):
        roi_enhanced_save_path = f'{self.bse_save_path}/{self.file_name_pref}-enhanced-roi.bmp'
        
        center_offset = (self.config.center_offset_x, self.config.center_offset_y)
        rect = (self.config.rect_left, self.config.rect_right)
        bar_height = self.config.scale_bar_height
        crop_radius = self.config.crop_radius
        downsample_img = self.crop_circle_with_bar(center_offset, rect, crop_radius, bar_height)
       
        roi_result = self.crop_roi_rect(downsample_img, 
                                        [self.config.rect_roi_width, self.config.rect_roi_height],
                                        self.config.roi_cropped_offset)
        clipLimit = self.config.clipLimit
        tileGridSize = (self.config.tileGridSize[0], self.config.tileGridSize[1])
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        clahe_image = clahe.apply(roi_result)

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
    
    # image是cv2读取的numpy对象
    def kms_segmentation(self, image):
        cls_num = self.config.class_num
        save_path = self.bse_save_path
        file_pref = self.file_name_pref

        for i in range(cls_num):
            classified_num = i + 2
            seg_result = self.segmentation.kmeans_image_segmentation(image, classified_num)
            classified_path = f"{save_path}/{file_pref}-enhanced-roi-kms{classified_num}.bmp"
            cv2.imwrite(classified_path, seg_result)
    
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

    def seg_and_crop_masked(self, image, final_res_suffix):
        # 1. 对图像进行分割
        self.kms_segmentation(image)
        path_pref = f"{self.bse_save_path}/{self.file_name_pref}"
        classfied_img_path = f"{path_pref}-enhanced-roi-kms{self.config.mask_classfied_num}.bmp"
        kms = cv2.imread(classfied_img_path, cv2.IMREAD_GRAYSCALE)
        # 2. 经过一些腐蚀操作去除掉一些细微的颗粒
        processed_img = self.segmentation.morphy_process_kms_image(kms, self.config.gray_cls, self.config.kernel_size)
        cv2.imwrite(f'{path_pref}-enhanced-roi-kms3-filter.bmp', processed_img)
        # 3. 计算联通区域筛选出较大的颗粒
        filterred_image = self.segmentation.filter_small_size_out(processed_img, self.config.size_threshold)
        cv2.imwrite(f"{path_pref}-masked.bmp", filterred_image)

        return self.crop_processed_bse_bin_images(
            image, filterred_image, 
            [self.config.start_left, self.config.start_top],
            [self.config.cropped_width, self.config.cropped_height],
            [self.config.offset_x, self.config.offset_y],
            final_res_suffix)
    
    def total_matched_img_processed(self):
        # 裁剪并增强
        roi_enhanced = self.crop_enhaned_bse()
        # 分割并保存
        self.seg_and_crop_masked(roi_enhanced)
