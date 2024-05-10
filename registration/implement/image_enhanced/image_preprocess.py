import cv2, os
import numpy as np
import matplotlib.pyplot as plt

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

        self.clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))

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
        enhanced_ct = self.clahe.apply(ct_img)
        return enhanced_ct

    def segment_ct(self, ct_img_enhanced, cls_num=4, random = None):
        return self.segmentation.kmeans_image_segmentation(ct_img_enhanced, cls_num, random)

    def filter_segment_img(self, img, cls_intensity, particle_size):
        pass

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
        return mi

    # 正向或者反向获取最优的slice
    def get_best_slice_idx(self, start_idx, interval, refer_img, crop_rect, rot, forward = True):
        delta = -1
        if forward : delta = 1

        start_ct_img = Tools.get_ct_img(self.config.cement_sample_index, start_idx)
        best_slice_index = start_idx
        best_mi = self.compute_mi_in_cropped(start_ct_img, refer_img, crop_rect, rot)

        for i in range(interval):
            slice_idx = start_idx + (i + 1)*delta
            ct_img = Tools.get_ct_img(self.config.cement_sample_index, slice_idx)

            mi = self.compute_mi_in_cropped(ct_img, refer_img, crop_rect, rot)
            print(f"slice: {slice_idx}, mi: {mi}")
            if mi > best_mi :
                best_mi = mi
                best_slice_index = slice_idx

        return best_mi, best_slice_index


    # 匹配图像成功后最后的一些处理
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

    def get_best_slice_idx_from_ct(self, init_slice_index, bse_cropped_matched, cropped_rect):
        latent_slice_area = self.config.latent_slice_area

        ct_size = self.config.ground_ct_size
        ct_img = Tools.get_ct_img(self.config.cement_sample_index, init_slice_index)
        ct_cropped_matched = Tools.crop_rotate_mi(ct_img, 
                                   [ct_size[0] * 0.5, ct_size[0] * 0.5],
                                   ct_size,
                                   self.config.matched_rotation,
                                   cropped_rect
                                   )

        init_mi = Tools.mutual_information(bse_cropped_matched, ct_cropped_matched)
        print(f"init best slice: {init_slice_index}, mi: {init_mi}")

        forward_best_mi, forward_best_idx = self.get_best_slice_idx(init_slice_index, 
                                                               latent_slice_area, 
                                                               bse_cropped_matched, 
                                                               cropped_rect, 
                                                               self.config.matched_rotation)
        backward_best_mi, backward_best_idx = self.get_best_slice_idx(init_slice_index, 
                                                               latent_slice_area, 
                                                               bse_cropped_matched, 
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

        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))

        for i in range(interval):
            index = slice_index + i
            ct_img = Tools.get_ct_img(self.config.cement_sample_index, index)
            file_name = f"{index}_enhanced.bmp"
            # 进行旋转
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)     
            rotated_image = cv2.warpAffine(ct_img, rotation_matrix, (ct_size[0], ct_size[1]))
            # 进行对比度增强
            enhanced_img = clahe.apply(rotated_image)
            # 进行保存
            Tools.save_img(file_path, file_name, enhanced_img)


    # 选择最佳潜在区域，并保存起来
    def choose_best_slices_save(self):
        # (1) 剪切区域重映射
        # (2) 获取匹配区域最优的slice
        # (3) 获取整个BSE ROI最优的slice
        # (4) 对最优的BSE进行选择
        # (5) 得到最优区域之后将图像复制到指定文件夹中
        matched_rect_init, matched_rect_bse = self.cropped_rect_remapping()
        matched_bse_refer_path, matched_bse_refer_filename = Tools.get_processed_referred_path(self.config)
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
