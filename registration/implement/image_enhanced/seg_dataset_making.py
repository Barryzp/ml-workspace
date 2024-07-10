import cv2, os
import numpy as np

from utils.tools import Tools
from utils.segmentation_kms import SegmentationKMS
from utils.common_config import CommonConfig
from utils.visualize import VisualizeData

# 制作水泥分割图像数据集
class CementSegDatasetMaking:
    
    def __init__(self, config) -> None:
        self.config = config

        self.ct_ori_save_path = f"{self.config.data_save_root}/cement_dataset_seg/imgs"
        self.ct_mask_save_path = f"{self.config.data_save_root}/cement_dataset_seg/masks"

        self.segmentation = SegmentationKMS()

    def segment_ct(self, ct_img_enhanced, cls_num=4, random = None):
        return self.segmentation.kmeans_image_segmentation(ct_img_enhanced, cls_num, random)

    # 主要是用于来确定分割的水泥颗粒的灰度
    def show_seg_result(self):
        cement_id = self.config.cement_sample_index
        sample_range = CommonConfig.get_range(cement_id)
        first_slice = sample_range[0]

        kmeans_random = self.config.kmeans_random_status
        ct_seg_cls = self.config.mask_classfied_num

        img_ori = CommonConfig.get_cement_ct_slice(cement_id, first_slice)
        cls_ct, bin_ct = self.segment_ct(img_ori, ct_seg_cls, kmeans_random)

        # 可视化
        VisualizeData.show_imgs([img_ori, cls_ct], ["ori_ct", "seg_result"])

    # 分割水泥的未水化水泥颗粒
    def seg_cement_unhydrated_particale(self):

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
        ct_seg_cls = self.config.mask_classfied_num

        jpeg_quality = self.config.jpeg_qulity

        temp_mask_img = None
        # 记录所有的编号
        for i in range(loop_times):
            slice_index = start_index + i * sample_interval
            ori_ct_img = CommonConfig.get_cement_ct_slice(cement_id, slice_index)
            
            img_prefix = f"{cement_id}_{slice_index}"

            ori_ct_img_name = f"{img_prefix}.jpg"
            save_bin_img_name = f"{img_prefix}_mask.jpg"
            save_test_temp_img_name = f"{img_prefix}_temp.jpg"

            # 对比度增强
            if self.config.enhanced : enhanced_ct = self.enhanced_ct(ori_ct_img)
            enhanced_ct = ori_ct_img
            # 分割图像
            cls_ct, bin_ct = self.segment_ct(enhanced_ct, ct_seg_cls, kmeans_random)
            
            # HACK 过时方法，现在可以直接得到最大聚类中心，那个中心就是未水化的水泥颗粒
            # if temp_mask_img is not None:
            #     if self.config.save_temp_res : Tools.save_img(self.ct_mask_save_path, save_test_temp_img_name, cls_ct)
            #     ct_gray_cls = Tools.bin_mask_mode_gray_cls_10(temp_mask_img, cls_ct)
            
            # 二值化图像
            ct_bin_img = bin_ct #Tools.binarized_img(cls_ct, ct_gray_cls)
            # 2. 计算联通区域筛选出较大的颗粒
            filterred_image = self.segmentation.filter_small_size_out(ct_bin_img, self.config.size_threshold)
            # 3. 经过一些腐蚀操作去除掉一些细微的颗粒
            processed_img = self.segmentation.morphy_process_kms_image(filterred_image, self.config.kernel_size)
            # (1) 存储遮罩图像
            Tools.save_img_jpg(self.ct_mask_save_path, save_bin_img_name, processed_img, jpeg_quality)
            # (2) 存储原图
            Tools.save_img_jpg(self.ct_ori_save_path, ori_ct_img_name, ori_ct_img, jpeg_quality)
            temp_mask_img = processed_img