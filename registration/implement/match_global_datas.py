import torch, cv2
import numpy as np
from utils.tools import Tools

class GlobalMatchDatas():

    def __init__(self, config, reg) -> None:
        self.aim_slice_index = -1
        self.global_best_value = -1000
        self.iteration_count = 0
        self.lower_save_count = 0
        self.global_best_position = torch.tensor([.0, .0, .0])
        self.global_best_img = None
        self.share_records_out = []
        self.threshold = config.matched_threshold
        
        self.stop_loop = False
        self.config = config
        self.reg_obj = reg

    # 这个best_val都是正值
    def set_best(self, best_val, best_position, best_img, ct_slice_index):
        if best_val > self.global_best_value:
            self.iteration_count += 1
            self.global_best_value = best_val
            self.global_best_position = best_position
            self.global_best_img = best_img
            print(f"id: {self.iteration_count}; best val: {best_val}; ct_slice_index: {ct_slice_index}")
            
            ori_slice_img, ct_slice_index = self.reg_obj.crop_slice_from_ori_3dct(best_position)
            self.aim_slice_index = ct_slice_index

            # 把这张图片保存一下
            file_path = Tools.get_save_path(self.config)
            mask_file_name = f"{self.iteration_count}-{ct_slice_index}-a-mask_ct.bmp"
            Tools.save_img(file_path, mask_file_name, best_img)
            slice_file_name = f"{self.iteration_count}-{ct_slice_index}-a-slice_ct.bmp"
            Tools.save_img(file_path, slice_file_name, ori_slice_img)
            
            # self.save_best_match(ct_slice_index, best_position)

        # if best_val > self.config.matched_save_lower_threshold:
        #     # 临时保存一下
        #     self.lower_save_count += 1
        #     self.save_above_crop_ct(ct_slice_index, best_position)

        if self.global_best_value > self.threshold:
            self.stop_loop = True

    def save_above_crop_ct(self, slice_index, position):
        ct_ori_file_name = f"{self.lower_save_count}_{slice_index}_match_ori_ct.bmp"
        ct_mask_file_name = f"{self.lower_save_count}_{slice_index}_match_mask_ct.bmp"
        self.save_match_ct(slice_index, position, ct_ori_file_name, ct_mask_file_name)

    def save_match_ct(self, slice_index, position, ct_ori_file_name, ct_mask_file_name):
        file_path = Tools.get_save_path(self.config)
        # 另外，并保存原始CT图像和maskCT，这个是原始大小，如果完全加载进来就会太大，先不这样
        crop_x, crop_y = position[0].item(), position[1].item()
        rot = position[-1].item()
        downsample_times = self.config.downsample_times
        height, width = self.reg_obj.get_bse_img_shape()
        rect = np.array([crop_x, crop_y, width, height]) * downsample_times
        ct_src = f"{self.config.data_path}/sample{self.config.cement_sample_index}/ct/matched"
        ct_ori_name = f"{slice_index}_enhanced_ct.bmp"
        ct_mask_name = f"{slice_index}_mask_ct.bmp"
        ct_ori = cv2.imread(f"{ct_src}/{ct_ori_name}", cv2.IMREAD_GRAYSCALE)
        ct_mask = cv2.imread(f"{ct_src}/{ct_mask_name}", cv2.IMREAD_GRAYSCALE)
        # 1. 获取在原始大小遮罩CT图像的mask结果
        result_mask_ct_matched = Tools.rotate_and_crop_img(ct_mask, rot, rect)
        # 2. 获取在原始大小下CT图像的结果
        result_ct_matched = Tools.rotate_and_crop_img(ct_ori, rot, rect)
        Tools.save_img(file_path, ct_ori_file_name, result_ct_matched)
        Tools.save_img(file_path, ct_mask_file_name, result_mask_ct_matched)

    def save_best_match(self, slice_index, position, random_id = ""):
        ct_ori_file_name = f"{self.iteration_count}_best_match_ori_ct_{slice_index}.bmp"
        ct_mask_file_name = f"{self.iteration_count}_best_match_mask_ct_{slice_index}.bmp"
        self.save_match_ct(slice_index, position, ct_ori_file_name, ct_mask_file_name)

    # 保存最佳图像 并截取对应剪切的CT图像
    def save_all_best_match_imgs(self):
        file_path = Tools.get_save_path(self.config)
        best_slice = self.aim_slice_index

        mask_file_name = f"1A-{best_slice}-a-mask_ct.bmp"
        bse_mask_name = f"1A-{best_slice}-a-mask_bse.bmp"
        ct_file_name = f"1A-{best_slice}-b-ori_ct.bmp"
        bse_file_name = f"1A-{best_slice}-b-ori_bse.bmp"

        position = self.global_best_position
        crop_x, crop_y = position[0].item(), position[1].item()
        rot = position[-1].item()

        downsample_times = self.config.downsample_times
        height, width = self.reg_obj.get_bse_img_shape()
        rect = np.array([crop_x, crop_y, width, height]) * downsample_times

        # 1. 原本的bse图像
        bse_img_ori = self.reg_obj.bse_img_ori
        bse_mask_ori = self.reg_obj.bse_mask_ori
        # 2. 原本的bse遮罩图像
        # 加载ct图像和ct遮罩图像
        ct_src = f"{self.config.data_path}/sample{self.config.cement_sample_index}/ct/matched"
        ct_ori_name = f"{self.aim_slice_index}_enhanced_ct.bmp"
        ct_mask_name = f"{self.aim_slice_index}_{self.config.ct_mask_suffix}.bmp"
        ct_ori = cv2.imread(f"{ct_src}/{ct_ori_name}", cv2.IMREAD_GRAYSCALE)
        ct_mask = cv2.imread(f"{ct_src}/{ct_mask_name}", cv2.IMREAD_GRAYSCALE)
        # 1. 获取在原始大小遮罩CT图像的mask结果
        result_mask_ct_matched = Tools.rotate_and_crop_img(ct_mask, rot, rect)
        # 2. 获取在原始大小下CT图像的结果
        result_ct_matched = Tools.rotate_and_crop_img(ct_ori, rot, rect)
        # 3. 原本的CT最优匹配裁剪区域图像
        # 4. CT最优匹配裁剪区域mask图像
        Tools.save_img(file_path, ct_file_name, result_ct_matched)
        Tools.save_img(file_path, mask_file_name, result_mask_ct_matched)
        Tools.save_img(file_path, bse_file_name, bse_img_ori)
        Tools.save_img(file_path, bse_mask_name, bse_mask_ori)

    def get_loop_state(self):
        return self.stop_loop
    
    def put_in_share_objects(self, ls):
        self.share_records_out.append(ls)