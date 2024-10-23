import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from functools import partial
from utils.tools import Tools
from enums.global_var import MatchUnit
from dtos.pyramid_reg import PyramidCfg

class Registration:
    def __init__(self, config, ct_slice_inteval = None, ct_index_array = None) -> None:
        # 过时了，用的itk库
        self.itk_img = None
        
        # fine registration, 精配准
        self.reg_3dct = {}
        # 对应配置
        self.fine_reg_pyramids = {}

        # ct图像3d，这是一个数组噢
        self.ct_matched_3d = []
        # ctmask图像3d
        self.ct_matched_msk_3d = []

        self.bse_img = None
        self.ct_img = None
        # 老早的版本了，用于mi + sp
        self.masked_img = None

        # region 未经过下采样的原始图像
        # 这两个是bse图像中的
        self.bse_img_ori = None
        self.bse_mask_ori = None
        self.bse_ori_shape = None
        # bse图像位于ct图像中的索引，注意：这个值只是一个相对值
        self.bse_indeces_in_ct = None
        # 这两个是ct图像中的，在匹配的时候是为空的
        self.ct_img_ori = None
        self.ct_msk_img_ori = None
        self.ct_slice_ori_shape = None
        # endregion

        self.optim_framework = None
        self.config = config

        if self.config.mode == "matched":
            # 匹配过程中ct的索引数组
            self.ct_index_array = ct_index_array
            # 下采样的切片间隔
            self.bse_diameter_interval = None
            self.ct_slice_inteval = ct_slice_inteval
            self.start_ct_index = ct_index_array[0]
            self.matched_3dct_indeces = None
            self.matched_ct_imgs = {}
            self.set_matched_3dct_indeces()

        self.load_img()
        self.set_config_delta()

    def set_config_delta(self):
        mode = self.config.mode
        if mode == "2d" or mode == "2d-only":
            bse_height, bse_width = self.get_bse_img_shape()
            ct_height, ct_width = self.config.cropped_ct_size[0], self.config.cropped_ct_size[1]
            self.config.rotation_center_xy = [ct_width/2, ct_height/2]
            translate_x = ct_width - bse_width
            translate_y = ct_height - bse_height
            self.config.translate_delta[0] = translate_x
            self.config.translate_delta[1] = translate_y

    def get_ct_index_array(self):
        return self.ct_index_array

    def get_3dct_index_array(self, index):
        return self.matched_3dct_indeces[index]

    def get_matched_3dct_indeces(self):
        return self.matched_3dct_indeces

    def get_matched_3d_msk_ct(self, matched_index):
        return self.ct_matched_msk_3d[matched_index]

    def get_matched_3d_ct(self, matched_index):
        return self.ct_matched_3d[matched_index]

    # 设置匹配时的分块三维ct图像的切片索引
    def set_matched_3dct_indeces(self):
        # 作为一个整体加入进去
        if self.config.match_unit == MatchUnit.comp_one:
            self.matched_3dct_indeces = [self.ct_index_array]
            return

        downsample3dct_slices = int(self.config.matched_3dct_depth / self.config.downsample_times)
        ct_index_array = self.ct_index_array

        # 以防最优解可能在划分的切片之间
        ct3d_interval = downsample3dct_slices // 2
        ct3d_nums = len(ct_index_array) // ct3d_interval
        # 余出来的完全可以单独地干，因为必然是大于切片数量是大于10的，
        # 而我们要求的最低CT高度为ceil(depth/downsample_times)，换算也就是ceil(20/8) = 3张，咋个都可以了
        index_array = None

        ct_3d_indeces = []
        for i in range(ct3d_nums):
            ct_start_index = i * ct3d_interval
            ct_end_index = ct_start_index + downsample3dct_slices
            # 切分的ct块的索引数组
            if i == ct3d_nums - 1: index_array = ct_index_array[ct_start_index:]
            else: index_array = ct_index_array[ct_start_index:ct_end_index]
            ct_3d_indeces.append(index_array)
        self.matched_3dct_indeces = ct_3d_indeces

    def set_optim_algorithm(self, optim):
        self.optim_framework = optim
        # 需要绑定实例对象
        similarity_fun = partial(self.similarity)
        if self.config.mode == "3d":
            similarity_fun = partial(self.similarity, pyramid = optim.pyramid_cfg)
        optim.set_init_params(similarity_fun, self)

    def get_bse_diameter_interval(self):
        return self.bse_diameter_interval
        sample_id = self.config.cement_sample_index
        zoom_times = self.config.bse_zoom_times
        times = zoom_times // 100
        suffix = self.config.zoom_bse_index
        self.file_name_pref = f"{sample_id}-{times}-{suffix}"
        bse_save_path = f"{self.config.data_path}/sample{sample_id}/bse/s{self.config.sample_bse_index}/{zoom_times}"

        config_name = f"{self.file_name_pref}-config.yaml"
        bse_config = Tools.load_yaml_config(f"{bse_save_path}/{config_name}")

        return [bse_config.size_threshold_bse_min, bse_config.size_threshold_bse_max]

    def _load_matched_ct3d(self, ct_index_array):
        data_path = self.config.data_path
        cement_sample_index = self.config.cement_sample_index
        ct_image_path = f"{data_path}/sample{cement_sample_index}/ct/matched"

        scale_ratio = self.config.size_threshold_ratio
        diamter_interval = self.get_bse_diameter_interval()
        particle_min = diamter_interval[0]
        particle_max = diamter_interval[1] * scale_ratio

        ct_matched_msk_3d = None
        ct_matched_3d = None

        for index in ct_index_array:
            mask_file_path = f"{ct_image_path}/{index}_{self.config.ct_mask_suffix}.bmp"
            img_file_path = f"{ct_image_path}/{index}_{self.config.ct_slice_suffix}.bmp"

            ct_slice_img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE) 
            ct_mask_img = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
            self.ct_slice_ori_shape = ct_mask_img.shape

            filtered_toggle = self.config.filtered_toggle

            # 用于兼容以前的代码，以前只过滤了较小粒径的水泥颗粒，这样灵活一些
            ct_mask_img = Tools.filter_by_diameter_bin_img(ct_mask_img, [particle_min, particle_max]) if filtered_toggle else ct_mask_img

            if self.config.downsampled: 
                ct_mask_img = Tools.downsample_bin_img(ct_mask_img, self.config.downsample_times)
                ct_slice_img = Tools.downsample_image(ct_slice_img, self.config.downsample_times)
            
            # 这个属性没啥作用
            self.matched_ct_imgs[index] = ct_mask_img
            if self.config.mode == "matched": self.config.cropped_ct_size = ct_mask_img.shape

            # 升维，作为三维图像的一个切面
            slice_msk_img = ct_mask_img[np.newaxis, :]
            slice_ct_img = ct_slice_img[np.newaxis, :]
            if ct_matched_msk_3d is None :
                ct_matched_msk_3d = slice_msk_img
                ct_matched_3d = slice_ct_img
            else: 
                ct_matched_msk_3d = np.concatenate([ct_matched_msk_3d, slice_msk_img], axis=0)
                ct_matched_3d = np.concatenate([ct_matched_3d, slice_ct_img], axis=0)
        
        return ct_matched_3d, ct_matched_msk_3d

    def _load_matched_ct3d_imgs(self):
        cts_index_array = self.matched_3dct_indeces

        for ct_index_array in cts_index_array:
            ct_matched_3d, ct_matched_msk_3d = self._load_matched_ct3d(ct_index_array)
            self.ct_matched_3d.append(ct_matched_3d)
            self.ct_matched_msk_3d.append(ct_matched_msk_3d)

    def _load_ct_img_debug(self):
        self.ct_img = cv2.imread(self.config.debug_ct_path, cv2.IMREAD_GRAYSCALE)
        self.ct_msk_img_ori = np.copy(self.ct_img)
        self.ct_img_ori = cv2.imread(self.config.debug_ct_ori_path, cv2.IMREAD_GRAYSCALE)
        if self.config.downsampled: self.ct_img = Tools.downsample_image(self.ct_img, self.config.downsample_times)
        self.config.cropped_ct_size = self.ct_img.shape

    def _load_ct_img(self):
        data_path = self.config.data_path
        cement_sample_index = self.config.cement_sample_index
        sample_bse_index = self.config.sample_bse_index
        ct_2d_index = self.config.ct_2d_index

        scale_ratio = self.config.size_threshold_ratio
        diamter_interval = self.get_bse_diameter_interval()
        particle_min = diamter_interval[0]
        particle_max = diamter_interval[1] * scale_ratio

        ct_image_path = f"{data_path}/sample{cement_sample_index}/ct/matched"
        self.ct_img = cv2.imread(f"{ct_image_path}/{ct_2d_index}_mask_ct.bmp", cv2.IMREAD_GRAYSCALE)
        self.ct_img_ori = cv2.imread(f"{ct_image_path}/{ct_2d_index}_enhanced_ct.bmp", cv2.IMREAD_GRAYSCALE)

        self.ct_img = Tools.filter_by_diameter_bin_img(self.ct_img, [particle_min, particle_max]) if self.config.filtered_toggle else self.ct_img
        Tools.save_img(ct_image_path, "test_filter_ct.bmp", self.ct_img)

        if self.config.downsampled: self.ct_img = Tools.downsample_image(self.ct_img, self.config.downsample_times)
        self.config.cropped_ct_size = self.ct_img.shape

    def get_ct_img_shape(self):
        # (height, width)(rows, column)
        return self.ct_img.shape

    # 加载精配准的ct图像
    def _load_fine_reg_ct(self):
        # 由于要进行金字塔匹配，因此需要加载不同重采样下的图像
        match_downsamples = self.config.downsample_times
        max_downsample_pow = int(np.log2(match_downsamples))
        
        ct3d_intervals = []
        for ds_pow in range(max_downsample_pow-1, -1, -1):
            load_interval = 2**ds_pow
            ct3d_intervals.insert(0, load_interval)
            self.reg_3dct.setdefault(load_interval, [])

        best_matched_slice = self.config.matched_slice_index
        half_latent_depth = int(self.config.latent_depth_area * 0.5)
        start_idx = best_matched_slice - half_latent_depth
        end_idx = best_matched_slice + half_latent_depth

        data_path = self.config.data_path
        cement_sample_index = self.config.cement_sample_index
        ct_path = f"{data_path}/sample{cement_sample_index}/ct/s{self.config.sample_bse_index}"
        counter = 0
        for slice_idx in range(start_idx, end_idx, 1):
            ct_slice_path = f"{ct_path}/slice_{slice_idx}.bmp"
            if self.config.enhanced : ct_slice_path = f"{ct_path}/{slice_idx}_enhanced.bmp"
            ct_slice_img = cv2.imread(ct_slice_path, cv2.IMREAD_GRAYSCALE) 

            # 叠上去，实际上这个interval就是downsample_times（下采样倍数），之后再叠在一块儿
            for interval in ct3d_intervals:
                if counter % interval == 0:
                    ds_img = Tools.downsample_image(ct_slice_img, interval, interpolation=cv2.INTER_NEAREST)
                    self.reg_3dct.get(interval).append(ds_img)
            counter += 1

        # 获得了完整的3d ct
        for interval in self.reg_3dct:
            ct3d = self.reg_3dct[interval]
            self.reg_3dct[interval] = np.stack(ct3d, axis=0)

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
    
    def _load_bse_img_debug(self):
        self.bse_img = cv2.imread(self.config.debug_bse_path, cv2.IMREAD_GRAYSCALE)
        self.bse_img_ori = cv2.imread(self.config.debug_bse_ori_path, cv2.IMREAD_GRAYSCALE)
        self.bse_mask_ori = np.copy(self.bse_img)
        if self.config.downsampled: 
            self.bse_img = Tools.downsample_image(self.bse_img, self.config.downsample_times)

    # 加载参考图像
    def _load_bse_img(self):
        src_path, file_name = Tools.get_processed_bse_path(self.config)
        prefix = file_name
        suffix = self.config.bse_suffix

        bse_file_name = f"{prefix}-{suffix}"
        bse_file_path = f'{src_path}/{bse_file_name}.bmp'
        self.bse_img = cv2.imread(bse_file_path, cv2.IMREAD_GRAYSCALE)
        self.bse_img_ori = np.copy(self.bse_img)
        self.bse_ori_shape = self.bse_img_ori.shape

        mode = self.config.mode

        # HACK 暂时先这么处理吧
        if mode == "matched" or mode == "2d-only":
            bse_mask_path = f"{src_path}/{prefix}-{self.config.mask_suffix}.bmp"
            mask_img = cv2.imread(bse_mask_path, cv2.IMREAD_GRAYSCALE)
            self.bse_mask_ori = np.copy(mask_img)
            # 对粒径进行处理
            # 首先获取有多少的颗粒
            # HACK 注意保存粒径，现在是在测试阶段
            _, contours = Tools.find_contours_in_bin_img(mask_img)
            diameter_interval = Tools.get_typical_particle_diameter(contours, self.config.quantile)
            mask_img = Tools.filter_by_diameter_bin_img(mask_img, diameter_interval) if self.config.filtered_toggle else mask_img
            # Tools.save_img(src_path, "test_filter.bmp", mask_img)
            self.bse_img = mask_img
            self.bse_diameter_interval = diameter_interval

        # 金字塔精配准就不进行下采样了

        if self.config.downsampled: self.bse_img = Tools.downsample_image(self.bse_img, self.config.downsample_times)
        r_img_height, r_img_width = self.bse_img.shape
        print(f"r_width: {r_img_width}, r_height: {r_img_height}")

    def get_bse_img_shape(self):
        # (height, width)(rows, column)
        return self.bse_img.shape

    def get_bse_ori_img_shape(self):
        # (height, width)(rows, column)
        return self.bse_img_ori.shape

    # 加载图像
    def load_img(self):
        self._load_bse_img()
        mode = self.config.mode

        if mode == "2d" or mode == "2d-only":
            self._load_ct_img()
        elif mode == "3d":
            self._load_fine_reg_ct()
        elif mode == "matched":
            self._load_matched_ct3d_imgs()
            bse_height, bse_width = self.get_bse_img_shape()
            self.bse_indeces_in_ct = self.init_bse_indeces((bse_width, bse_height))

    # 计算距离切片的偏移, size:[width, height]
    def compute_offset_from_ct(self, ct_size, slice_size):
        delta_x = ct_size[0] * .5 - slice_size[0] * .5
        delta_y = ct_size[1] * .5 - slice_size[1] * .5
        return [delta_x, delta_y]

    # 初始化好bse位于ct图像中的索引
    def init_bse_indeces(self, bse_img_size):
        bse_width, bse_height = bse_img_size

        # 这个还是不加上去，只作为参考的初始索引，左上角为原点，变换时索引则为：coordinates + optimized_position
        # bse_offset_from_origin = self.compute_offset_from_ct([ct_slice_width, ct_slice_height],
        #                                                      [bse_width, bse_height])
        # 使用 meshgrid 生成网格，然后将其转换为所需格式
        x, y = np.meshgrid(np.arange(bse_height), np.arange(bse_width))
        # 转置网格坐标，使其符合 [N, M] 的形状
        x = x.T
        y = y.T
        # 堆叠坐标网格形成所需的数组
        coordinates = np.stack((x, y), axis=-1)
        # 创建一个与 coordinates 形状匹配的常量数组
        constant_array = np.full(coordinates.shape[:-1] + (1,), 0)

        # 将 coordinates 和 constant_array 连接起来
        return np.concatenate((coordinates, constant_array), axis=-1)

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

    # 匹配切片相关
    def crop_slice_from_mask_3dct(self, x, volume_index,interpolation = "None"):
        volume = self.get_matched_3d_msk_ct(volume_index)
        return self.crop_slice_from_3dct_match(x, volume_index, volume, interpolation)

    def crop_slice_from_ori_3dct(self, x, volume_index, interpolation = "None"):
        volume = self.get_matched_3d_ct(volume_index)
        return self.crop_slice_from_3dct_match(x, volume_index, volume, interpolation)

    def crop_slice_from_3dct_match(self, x, volume_index, volume, interpolation = "None"):
        ct_index_array = self.get_3dct_index_array(volume_index)
        
        position = np.copy(x)

        translation = position[:3]
        rotation = position[3:]

        # 这个地方的bse_indeces_in_ct是不变的
        indeces_in_ct = Tools.translate_points(translation, self.bse_indeces_in_ct)
        height, width = self.get_bse_img_shape()
        # 围绕切片中心旋转
        rotation_center = indeces_in_ct[height // 2, width // 2]
        index_array = Tools.rotate_points(rotation_center, rotation, indeces_in_ct)
        integer_indeces = Tools.force_convert_uint(index_array)

        max_item = np.max(integer_indeces)
        # if max_item >= 240:
        #     print("stop here...")

        # 通过index_array来进行索引切片
        result =  Tools.safe_indexing_volume(volume, integer_indeces) #volume[integer_indeces[..., 2], integer_indeces[..., 0], integer_indeces[..., 1]]
        
        # 获取这个切片中Z方向最多的元素，也就是找到起最多包含的断层面
        most_prob_slice_index = Tools.most_frequent_element(integer_indeces[:, 2])
        # 还需要做一个映射 HACK 有问题这里
        most_prob_slice_index = ct_index_array[0] + most_prob_slice_index * self.ct_slice_inteval

        # HACK 用于后面的插值处理，目前看来还是不这样搞
        if interpolation == "None":
            pass
        
        return result, most_prob_slice_index

    def crop_rect_from_2dct(self, x, interpolation = "None"):
        image = self.ct_img_ori
        r_height, r_width = self.get_bse_ori_img_shape()
        f_height, f_width = image.shape

        rotation_center_xy = [f_width * .5, f_height * .5]


        ds_times = self.config.downsample_times

        x_delta = int(x[0].item() * ds_times)
        y_delta = int(x[1].item() * ds_times)

        # 步骤 2: 旋转图像、裁剪图像
        angle = x[2].item()

        pos_x, pos_y, w, h = x_delta, y_delta, r_width, r_height  # 裁剪位置和大小
        cropped_image = Tools.crop_rotate_mi(image, 
                                             rotation_center_xy, 
                                          (f_width, f_height), 
                                          angle, 
                                          [pos_x, pos_y, w, h])

        return cropped_image

    # 从ct图像中根据索引切片
    def crop_slice_from_3dct(self, x, volume, interpolation = "None"):
        position = np.copy(x)
        translation = position[:3]
        rotation = position[3:]

        # 这个地方的bse_indeces_in_ct是不变的
        indeces_in_ct = Tools.translate_points(translation, self.bse_indeces_in_ct)
        height, width = self.get_bse_img_shape()
        # 围绕切片中心旋转
        rotation_center = indeces_in_ct[height // 2, width // 2]
        index_array = Tools.rotate_points(rotation_center, rotation, indeces_in_ct)
        integer_indeces = Tools.force_convert_uint(index_array)

        max_item = np.max(integer_indeces)
        # if max_item >= 240:
        #     print("stop here...")

        # 通过index_array来进行索引切片
        result = Tools.safe_indexing_volume(volume, integer_indeces) #volume[integer_indeces[..., 2], integer_indeces[..., 0], integer_indeces[..., 1]]
        
        # 获取这个切片中Z方向最多的元素，也就是找到起最多包含的断层面
        most_prob_slice_index = Tools.most_frequent_element(integer_indeces[:, 2])
        # 还需要做一个映射 HACK 有问题这里
        most_prob_slice_index = self.ct_index_array[0] + most_prob_slice_index * self.ct_slice_inteval

        # HACK 用于后面的插值处理，目前看来还是不这样搞
        if interpolation == "None":
            pass
        
        return result, most_prob_slice_index

    # 从ct图像中根据索引切片
    def crop_slice_from_3dct_fine_reg(self, x, pyramid, interpolation = "None"):
        volume = self.reg_3dct.get(pyramid.downsample_times)
        
        position = np.copy(x)
        translation = position[:3]
        rotation = position[3:]

        bse_indeces_in_ct = pyramid.reg_bse_indeces

        # 这个地方的bse_indeces_in_ct是不变的
        indeces_in_ct = Tools.translate_points(translation, bse_indeces_in_ct)
        height, width = pyramid.reg_bse_img_size[1], pyramid.reg_bse_img_size[0]
        # 围绕切片中心旋转
        rotation_center = indeces_in_ct[int(height // 2), int(width // 2)]
        index_array = Tools.rotate_points(rotation_center, rotation, indeces_in_ct)
        integer_indeces = Tools.force_convert_uint(index_array)

        # 通过index_array来进行索引切片
        result = Tools.safe_indexing_volume(volume, integer_indeces) #volume[integer_indeces[..., 2], integer_indeces[..., 0], integer_indeces[..., 1]]
        
        # 获取这个切片中Z方向最多的元素，也就是找到起最多包含的断层面
        most_prob_slice_index = Tools.most_frequent_element(integer_indeces[:, 2])
        # 还需要做一个映射 HACK 有问题这里
        most_prob_slice_index = pyramid.start_idx + most_prob_slice_index * pyramid.downsample_times

        # HACK 用于后面的插值处理，目前看来还是不这样搞
        if interpolation == "None":
            pass
        
        return result, most_prob_slice_index

    # 只利用遮罩的空间信息
    def similarity_only_mask_spatial(self, x, ct_matching_slice_index):
        rotation_center_xy = self.config.rotation_center_xy

        image = self.matched_ct_imgs[ct_matching_slice_index]
        r_height, r_width = self.get_bse_img_shape()
        f_height = self.config.cropped_ct_size[0]
        f_width = self.config.cropped_ct_size[1]

        angle = x[2].item()
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center_xy, angle, scale)     
        rotated_image = cv2.warpAffine(image, rotation_matrix, (f_width, f_height))     

        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]
        spatial_info = self.spatial_correlation_with_mask(cropped_image, self.bse_img)
        return spatial_info, cropped_image

    def similarity_matched_dice(self, x, match3dct_idx):
        cropped_image, latent_z = self.crop_slice_from_mask_3dct(x, match3dct_idx)
        dice = Tools.dice_coefficient(cropped_image, self.bse_img)
        return dice, cropped_image, latent_z, 0, 0, 0

    def similarity_3d(self, x, pyramid):
        cropped_img, latent_z = self.crop_slice_from_3dct_fine_reg(x, pyramid)
        bse_img = pyramid.bse_img
        mi = self.mutual_information(cropped_img, bse_img)
        return mi, cropped_img, latent_z

    def similarity_jaccard(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        image = self.ct_img
        r_height, r_width = self.get_bse_img_shape()
        f_height, f_width = self.get_ct_img_shape()

        # 步骤 2: 旋转图像、裁剪图像
        angle = x[2].item()

        pos_x, pos_y, w, h = int(x[0].item()), int(x[1].item()), r_width, r_height  # 裁剪位置和大小
        cropped_image = Tools.crop_rotate_mi(image, 
                                          rotation_center_xy, 
                                          (f_width, f_height), 
                                          angle, 
                                          [pos_x, pos_y, w, h])

        # 特殊处理得到的二值化图像，计算jaccard系数
        jaccard_value = Tools.jaccard_index(cropped_image, self.bse_img)
        return jaccard_value, cropped_image, 0, 0, 0

    # 用于匹配的dice系数比较相似度
    def similarity_dice(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        image = self.ct_img
        r_height, r_width = self.get_bse_img_shape()
        f_height, f_width = self.get_ct_img_shape()

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
        dice_value = Tools.dice_coefficient(cropped_image, self.bse_img)

        return dice_value, cropped_image, 0, 0, 0

    def similarity_2d(self, x):
        rotation_center_xy = self.config.rotation_center_xy
        lamda_mis = self.config.lamda_mis

        image = self.ct_img
        r_height, r_width = self.get_bse_img_shape()
        f_height, f_width = self.get_ct_img_shape()

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
        mi_value = self.mutual_information(cropped_image, self.bse_img)

        spation_info = self.spatial_correlation_with_mask(cropped_image, self.bse_img)
        weighted_sp = lamda_mis * spation_info
        mis = mi_value + weighted_sp
        return mis, cropped_image, mi_value, spation_info, weighted_sp

    def similarity(self, x, match3dct_idx=None, pyramid=None):
        if self.config.debug :
            # 反射机制：通过函数名字符串调用对象的成员方法
            return getattr(self, self.config.debug_simiarity)(x)
            # if self.config.debug_simiarity == "similarity_dice":
            #     return self.similarity_dice(x)
        
        mode = self.config.mode

        if mode == "2d":
            return self.similarity_2d(x)
        elif mode == "2d-only":
            return self.similarity_dice(x)
        elif mode == "3d":
            return self.similarity_3d(x, pyramid)
        elif mode == "matched":
            return self.similarity_matched_dice(x, match3dct_idx)

    def save_matched_result(self, position):
        # 对于重采样的重新处理，旋转角度不需要操作的
        crop_x, crop_y = position[0].item(), position[1].item()
        rot = position[-1].item()

        downsample_times = self.config.downsample_times
        height, width = self.get_bse_img_shape()
        rect = np.array([crop_x, crop_y, width, height]) * downsample_times
        # 1. 获取在原始大小遮罩CT图像的mask结果
        result_mask_ct_matched = Tools.rotate_and_crop_img(self.ct_msk_img_ori, rot, rect)
        # 2. 获取在原始大小下CT图像的结果
        result_ct_matched = Tools.rotate_and_crop_img(self.ct_img_ori, rot, rect)
        
        file_path = Tools.get_save_path(self.config)
        mask_file_name = f"1Aa-mask_ct.bmp"
        bse_mask_name = f"1Aa-mask_bse.bmp"
        ct_file_name = f"1Aa-ori_ct.bmp"
        bse_file_name = f"1Aa-ori_bse.bmp"

        Tools.save_img(file_path, ct_file_name, result_ct_matched)
        Tools.save_img(file_path, mask_file_name, result_mask_ct_matched)
        Tools.save_img(file_path, bse_file_name, self.bse_img_ori)
        Tools.save_img(file_path, bse_mask_name, self.bse_mask_ori)

    def registrate(self):
        fitness, best_reg, best_position = self.optim_framework.run()

        if self.config.mode == "2d" and self.config.debug:
            self.save_matched_result(best_position)
        return fitness, best_reg, best_position

    # 参数的重映射，主要还是对于位移，角度不存在的
    def params_remapping(self, pyramid_pre, pyramid_new):
        out_downsamples = pyramid_new.downsample_times

        pre_ds_times = pyramid_pre.downsample_times
        pre_start_idx = pyramid_pre.start_idx
        pre_translation = pyramid_pre.translation

        # HACK 有问题哦，best_matched_slice并不代表位于中间
        # 先转换到原始匹配的大CT块
        translation_in_ori = pre_translation * pre_ds_times
        # z轴上还要加上起始索引才得到原始大CT块的相对位置
        translation_in_ori[-1] = translation_in_ori[-1] + pre_start_idx
        best_matched_slice = self.config.matched_slice_index
        half_latent_depth = int(self.config.latent_depth_area * 0.5)
        new_start_idx = best_matched_slice - half_latent_depth

        # 转换到新的块中的索引, 始终都是固定的，100那层嘛反正
        translation_now = translation_in_ori.copy()
        translation_now[-1] = translation_now[-1] - new_start_idx
        translation_now = translation_now / out_downsamples
        pyramid_new.start_idx = new_start_idx
        pyramid_new.end_idx = new_start_idx + self.config.latent_depth_area
        pyramid_new.rotation = pyramid_pre.rotation
        pyramid_new.translation = translation_now

    def build_pyramid_cfgs(self):
        cement_id = self.config.cement_sample_index
        bse_index = self.config.sample_bse_index
        match_cfg_path = f"{self.config.data_path}/sample{cement_id}/ct/s{bse_index}/cement_{cement_id}_s{bse_index}.yaml"
        matched_cfg = Tools.load_yaml_config(match_cfg_path)

        match_downsamples = self.config.downsample_times
        max_downsample_pow = int(np.log2(match_downsamples))
        
        ct_size = np.array(self.config.ct_size)
        bse_size = np.array(self.config.bse_size)

        ori_bse_img = self.bse_img_ori

        pyramid_cfgs = []
        # 1. 构建每层金字塔的配置
        for ds_pow in range(max_downsample_pow, -1, -1):
            if ds_pow == 3 or ds_pow == 0:
                downsample_times = 2**ds_pow
                cur_bse_size = bse_size / downsample_times
                pyramid = PyramidCfg.buildPyramid(downsample_times, 
                                                  cur_bse_size, 
                                                  ct_size / downsample_times)
                bse_indeces = self.init_bse_indeces((cur_bse_size[0], cur_bse_size[1]))
                bse_img_dsp = Tools.downsample_image(ori_bse_img, downsample_times=downsample_times, interpolation=cv2.INTER_NEAREST)

                pyramid.reg_bse_indeces = bse_indeces
                pyramid.bse_img = bse_img_dsp
                pyramid.start_idx = self.config.matched_start_idx
                pyramid.end_idx = self.config.matched_end_idx
                pyramid.translation = np.array(matched_cfg.matched_translate)
                pyramid.rotation = np.array(matched_cfg.matched_rotation)
                pyramid.delta_translation = np.array(self.config.translate_delta)
                pyramid.delta_rotation = np.array(self.config.rotation_delta)
                pyramid_cfgs.append(pyramid)
        
        return pyramid_cfgs

    # 精配准
    def fine_registrate(self, optim_class):
        match_downsamples = self.config.downsample_times
        max_downsample_pow = int(np.log2(match_downsamples))
        pyramid_cfgs = self.build_pyramid_cfgs()
        self.fine_reg_pyramids = pyramid_cfgs

        translation_delta = np.array(self.config.translate_delta)
        rotation_delta = np.array(self.config.rotation_delta)

        for i in range(len(pyramid_cfgs)-1):
            current_pyramid = self.fine_reg_pyramids[i+1]
            last_pyramid = self.fine_reg_pyramids[i]
            self.params_remapping(last_pyramid, current_pyramid)
            current_pyramid.delta_translation = translation_delta * current_pyramid.downsample_times
            current_pyramid.delta_rotation = rotation_delta
            best_val, best_pos = self.run_fine_reg_optim(optim_class, current_pyramid)
            current_pyramid.translation = best_pos[:3]
            current_pyramid.rotation = best_pos[3:]
        # 2. 每层金字塔进行精配准
        # for ds_pow in range(max_downsample_pow - 1, -1, -1):
        #     # 两次循环的数据结构不一样。。。
        #     pre_ds_times = 2**(ds_pow+1)
        #     downsample_times = 2**ds_pow
        #     current_pyramid = pyramid_cfgs.get(downsample_times)
        #     last_pyramid = pyramid_cfgs.get(pre_ds_times)
        #     self.params_remapping(last_pyramid, current_pyramid)
        #     current_pyramid.delta_translation = translation_delta * downsample_times
        #     current_pyramid.delta_rotation = rotation_delta
        #     self.run_fine_reg_optim(optim_class, current_pyramid)

        # 3. 输出最后一层金字塔配准结果

    def run_fine_reg_optim(self, optim_class, pyramid):
        config = self.config
        run_times = config.run_times

        gbest_val = -1
        gbest_pos = None

        for i in range(run_times):
            print(f"current loop times: {i+1}")
            rand_seed = config.rand_seed + i
            # 每一次迭代随机种子+1，这样的方式保证结果的一致
            np.random.seed(rand_seed)
            # global_match_datas = GlobalMatchDatas(config, self)
            optim = optim_class(config)
            optim.set_pyramid(pyramid)
            optim.set_runid(i)
            self.set_optim_algorithm(optim)
            optim.run_fine_reg()
            best_solution, best_val = optim.best_solution, optim.best_value
            if best_val > gbest_val:
                gbest_val = best_val
                gbest_pos = best_solution
            print(f"best val:{gbest_val}, pos: {gbest_pos}")

        return gbest_val, gbest_pos

    def _read_iter_data(self, record_id):
        # 先构造dict
        csv_path = self.config.data_save_path
        run_times = self.config.run_times

        # 我还想得到最高fitness对应的相关数据
        data_item = []
        # 保存每次迭代的最佳参数
        max_params_per_run = []

        for i in range(run_times):
            file_path = f"{csv_path}/{record_id}/{i}_0_pso_params_3d_ct.csv"
            csv_datas = pd.read_csv(file_path)
            fes = csv_datas['iterations'].values
            fitness = csv_datas['fitness'].values
            data_item.append(fitness)

            last_row = csv_datas.iloc[-1]
            max_params_per_run.append(last_row.to_list())
        
        np_arr = np.stack(data_item)
        iter_best = np_arr[:, -1]
        mean_best_fit = np.mean(iter_best)
        std_best_fit = np.std(iter_best)
        med = np.median(np_arr, axis=0)
        mean_fes = np.mean(np_arr, axis=0)
        data_dict = {
            "mean_best" : mean_best_fit, 
            "std_best_fit" : std_best_fit, 
            "median_fes" : med,
            "mean_fes" : mean_fes, 
            "fes" : fes
        }
        
        # 对max_params_per_run进行排序
        max_params_per_run = sorted(max_params_per_run, key=lambda x: x[-1], reverse=True)
        best_matched_params = max_params_per_run[0]
        # 打印最佳参数
        print(f"best_parameters: {best_matched_params}")

        if self.config.show_log:
            print(f"{record_id}, mean: {mean_best_fit:.4e}, std: {std_best_fit:.4e}")
        return data_dict

    def read_iter_data(self, record_ids):
        total_data_rows = 0
        record_dict = {}
        for record_id in record_ids:
            data_dict = self._read_iter_data(record_id)
            record_dict.setdefault(record_id, data_dict)
            total_data_rows = len(data_dict["fes"])

        record_dict.setdefault("total_data_rows", total_data_rows)
        return record_dict
    
    # 展示收敛曲线，这个total_mark代表的是显示多少个mark，在折线上（展示的是均值），纵轴上是以log10的对数
    def show_mean_convergence_line(self, record_dict, record_ids, mark_label, total_mark = None):
        step = 1
        # 需要获取总的列数
        total_iters = record_dict["total_data_rows"]
        if total_mark != None:
            step = total_iters // total_mark

        j = 0
        for record_id in record_ids:
            data_item = record_dict[record_id]
            mean = data_item["mean_fes"]
            mean = mean[::step]
            fes = data_item["fes"]
            fes = fes[::step]
            plt.plot(fes, mean, label=mark_label[j],
                     color=self.config.colors[j], marker=self.config.markers[j])
            j+=1
            
        plt.xlabel('FEs')
        plt.ylabel('Fitness')
        plt.legend()
        # 显示图像
        plt.show()