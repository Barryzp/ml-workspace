import torch, cv2
import numpy as np
from utils.tools import Tools

class GlobalMatchDatas():

    def __init__(self, config, reg) -> None:
        self.aim_slice_index = -1
        self.global_best_value = -1000
        self.iteration_count = 0
        self.lower_save_count = 0
        self.global_best_volume_index = -1
        self.global_best_position = torch.tensor([.0, .0, .0])
        self.global_best_img = None
        self.share_records_out = []
        self.threshold = config.matched_threshold
        
        self.stop_loop = False
        self.config = config
        self.reg_obj = reg

    # 这个best_val都是正值
    def set_best(self, best_val, best_position, best_img, ct_slice_index, volume_index, run_id=-1):
        if best_val > self.global_best_value:
            self.iteration_count += 1
            self.global_best_value = best_val
            self.global_best_position = best_position
            self.global_best_img = best_img
            print(f"id: {self.iteration_count}; best val: {best_val}; ct_slice_index: {ct_slice_index}")

            mode = self.config.mode
            if mode == "matched":
                self.global_best_volume_index = volume_index
                ori_slice_img, ct_slice_index = self.reg_obj.crop_slice_from_ori_3dct(best_position, volume_index)
                self.aim_slice_index = ct_slice_index
                # 把这张图片保存一下 HACK 最好加一个run_times
                file_path = Tools.get_save_path(self.config)
                mask_file_name = f"{self.iteration_count}-{run_id}-{ct_slice_index}-a-mask_ct.bmp"
                Tools.save_img(file_path, mask_file_name, best_img)
                slice_file_name = f"{self.iteration_count}-{run_id}-{ct_slice_index}-a-slice_ct.bmp"
                Tools.save_img(file_path, slice_file_name, ori_slice_img)
                # self.save_best_match(ct_slice_index, best_position)
            elif mode == "2d" or mode == "2d-only":
                pass

        if self.global_best_value > self.threshold:
            self.stop_loop = True

    def does_found_better_match(self):
        return self.global_best_value >= self.config.matched_lower_threshold

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

    # 保存最佳图像这里需要将坐标映射到3D的CT图像以及3D的Mask中，
    def save_best_3dct_inmatch(self):
        best_pos = self.global_best_position
        downsample_times = self.config.downsample_times
        bse_height, bse_width = self.reg_obj.bse_ori_shape

        index_array = self.reg_obj.get_3dct_index_array(self.global_best_volume_index)

        # HACK 这个切片始终有问题，得给它重新设置一下
        # 之前的逻辑太复杂，我直接就这样，
        # （1）先直接进行变换
        # （2）获取到最小和最大的Z
        # （3）最小的Z对应的实际slice_index是多少呢？这个需要记录一下

        # 1. 首先是获得咱们这个ct的start_slice_idx
        start_index_in_match3dct = index_array[0]
        # 2. 坐标重映射，映射到在原始坐标空间中的变换参数
        pos_remapping = Tools.remapping_from_optim_pos(best_pos, 
                                                       start_index_in_match3dct,
                                                       downsample_times)

        # 3. 将BSE索引进行变换，再进行读取图像
        transformed_indeces = Tools.transformed_slice_indeces([bse_width, bse_height], pos_remapping)
        # 4. 获取Z的最小值和最大值，然后连续读取即可
        all_z = transformed_indeces[:, :, -1]
        min_z = np.min(all_z)
        max_z = np.max(all_z)

        # 获取索引数组
        index_array = np.arange(min_z, max_z + 1)
        # 5. 加载原始大小的3d CT图像
        ct_ori = Tools.load_3dct_as_np_array(self.config.cement_sample_index, index_array)
        # 6. 对于Mask图像就直接上采样二值化即可
        mask_upsample = Tools.upsample_bin_img(self.global_best_img, downsample_times)
        # 7. 将坐标变换到以最小Z为起点的原点
        transformed_indeces[:, :, -1] -= min_z
        # 8. 对原始图像裁剪保存
        slice_from_ct = Tools.crop_slice_from_volume_use_position(transformed_indeces, ct_ori)
        
        # 保存这两张图片
        self.save_best_result(slice_from_ct, mask_upsample)

    def save_best_result(self, result_ct_matched, result_mask_ct_matched):
        file_path = Tools.get_save_path(self.config)
        best_slice = self.aim_slice_index

        mask_file_name = f"1A-{best_slice}-a-mask_ct.bmp"
        bse_mask_name = f"1A-{best_slice}-a-mask_bse.bmp"
        bse_filter_mask_name = f"1A-{best_slice}-a-mask-filter_bse.bmp"
        ct_file_name = f"1A-{best_slice}-b-ori_ct.bmp"
        bse_file_name = f"1A-{best_slice}-b-ori_bse.bmp"

        # 原本的bse图像
        bse_img_ori = self.reg_obj.bse_img_ori
        bse_mask_ori = self.reg_obj.bse_mask_ori
        bse_mask_filter = self.reg_obj.bse_img

        Tools.save_img(file_path, ct_file_name, result_ct_matched)
        Tools.save_img(file_path, mask_file_name, result_mask_ct_matched)
        Tools.save_img(file_path, bse_file_name, bse_img_ori)
        Tools.save_img(file_path, bse_mask_name, bse_mask_ori)
        Tools.save_img(file_path, bse_filter_mask_name, bse_mask_filter)

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
    
    # 清空共享数据（多层匹配的）
    def clear_share_objects(self):
        self.share_records_out = []