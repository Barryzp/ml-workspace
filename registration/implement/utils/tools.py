import cv2, os
import numpy as np
import itk, yaml
import pandas as pd
from math import radians
from munch import Munch
from pathlib import Path
from PIL import Image, ImageDraw
from .common_config import CommonConfig


class Tools:
    # 加载yaml文件并转化成一个对象
    def load_yaml_config(path):
        with open(Path(path), 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        config = Munch(yaml_config)
        return config

    # 将配置保存
    def save_obj_yaml(folder_path, file_name, obj):
        # 检查一下
        file_path = Tools.check_file(folder_path, file_name)
        # 将数据保存为 YAML 文件，中文内容正常显示
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.dump(obj, file, allow_unicode=True)

    def check_file_exist(folder_path, file_name):
        file_path = os.path.join(folder_path, file_name)
        return os.path.exists(file_path)

    def check_file(folder_path, file_name):
        file_path = os.path.join(folder_path, file_name)
        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            # 如果文件夹不存在，创建它
            os.makedirs(folder_path)
        else:
            # # 如果文件夹存在，检查特定文件是否也存在
            # if os.path.exists(file_path):
            #     # 如果文件存在，抛出异常
            #     raise FileExistsError(f"The file '{file_path}' already exists.")
            # print(f"Folder '{folder_path}' already exists and the file '{file_name}' is not present.")
            pass
        return file_path
    
    def get_save_path(config):
        # record_id用于标注哪一个，就行了
        path_prefix = config.data_save_path
        path_prefix = f"{path_prefix}/{config.record_id}"
        return path_prefix

    def get_processed_referred_path(config):
        sample_index = config.cement_sample_index
        bse_sample_index = config.sample_bse_index

        zoom_times = config.bse_zoom_times
        middle = zoom_times // 100
        end_index = config.zoom_bse_index

        file_name = f"{sample_index}-{middle}-{end_index}"
        # BSE 图像裁剪以及对比度增强处理
        src_path = f"D:/workspace/ml-workspace/registration/datasets/sample{sample_index}/bse/s{bse_sample_index}/{zoom_times}"
        return src_path, file_name

    def get_ct_img(cement_id, slice_index):
        return CommonConfig.get_cement_ct_slice(cement_id, slice_index)

    # 保存图片
    def save_img(folder_path, file_name, img):
        # 检查一下
        file_path = Tools.check_file(folder_path, file_name)
        # 保存图像, 这个img是np数组类型的
        saved_img = Image.fromarray(img.astype(np.uint8))
        saved_img.save(file_path)

    # 将数据保存至dataframe
    def save_params2df(datas, columns, folder_path, file_name):
        file_path = Tools.check_file(folder_path, file_name)
        data_df = pd.DataFrame(datas, columns = columns)
        data_df.to_csv(file_path, index=False)

    def caculate_hist_entropy(hist):
        p = hist / hist.sum()
        p = p[p > 0] # 移除概率为0的项
        return -np.sum(p * np.log2(p))

    # 计算图片的熵
    def caculate_entropy(image):
        hist = np.histogram(image, 256)[0]
        return Tools.caculate_hist_entropy(hist)

    def caculate_joint_entropy(img1, img2):
        hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=256)
        return Tools.caculate_hist_entropy(hist_2d)

    # 下采样图像
    def downsample_image(image_np, downsample_times, size = None):
        # 使用缩放因子来减半图片尺寸
        resized_image = cv2.resize(image_np, dsize=size, fx=1/downsample_times, fy=1/downsample_times, interpolation=cv2.INTER_AREA)
        return resized_image

    # 剪切并旋转，这个就是图像的mi中的那个
    def crop_rotate_mi(image, center, size, angle, rect):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)     
        rotated_image = cv2.warpAffine(image, rotation_matrix, (size[0], size[1]))
        pos_x, pos_y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])  # 裁剪位置和大小
        cropped_image = rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]
        return cropped_image

    # 旋转并剪切图像, 图像是nparray类型，旋转中心为图像中心，此外，以图像左上角作为裁剪原点
    def rotate_and_crop_img(image, angle, rect):
        height, width = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((height/2, width/2), angle, 1.0)     
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        pos_x, pos_y, w, h = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])  # 裁剪位置和大小
        cropped_image = rotated_image[pos_y:pos_y+h, pos_x:pos_x+w]
        return cropped_image

    # image是np对象，围绕着中心旋转
    def crop_rotate(image, center, size, angle):
        """
        Crop and rotate a region from an image.

        :param image: Source image
        :param center: Tuple (x, y) - the center of the region to crop
        :param size: Tuple (width, height) - the size of the region to crop
        :param angle: Rotation angle in degrees
        :return: Cropped and rotated image region
        """

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform rotation
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # Calculate the coordinates of the top-left corner of the cropped region
        x = int(center[0] - size[0] / 2)
        y = int(center[1] - size[1] / 2)

        # Crop the region
        # 防止出现超出数组范围的情况

        # Define row and column ranges
        start_row, end_row = y, y + size[0]  # Example row range
        start_col, end_col = x, x + size[1]  # Example column range

        # Define the fill value for out-of-bounds indices
        fill_value = 0.0

        # 对超出区域的处理
        cropped = [[rotated[i][j] if i >= 0 and i < len(rotated) and j >= 0 and j < len(rotated[i]) else fill_value 
              for j in range(start_col, end_col)] 
             for i in range(start_row, end_row)]
        # cropped = rotated[y:y + size[1], x:x + size[0]]

        return cropped
    
    # image: PIL的Image对象
    def crop_circle(image, radius, center):
        """
        裁剪灰度图像中的圆形区域。

        参数:
        image_path: 图像的路径。
        radius: 圆的半径。
        center: 圆的中心点坐标（x, y）。如果为None，则使用图像中心。
        """

        width, height = image.size

        # 创建遮罩
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius), fill=255)

        # 应用遮罩
        result = Image.new('L', (width, height))
        result.paste(image, (0, 0), mask)

        return result

    # image: PIL的Image对象
    def crop_rectangle(img, crop_rectangle):
        # 裁剪图像
        cropped_image = img.crop(crop_rectangle)
        return cropped_image

    # 从itk体素中进行切片（旋转中心，旋转角度，切片的索引）
    def get_slice_from_itk(itk_img, rotation_center, rotation, slice_indeces, size):
        # 定义平移和旋转参数
        translation_to_origin = itk.TranslationTransform[itk.D, 3].New()
        translation_to_origin.SetOffset(rotation_center)

        rotation_transform = itk.Euler3DTransform.New()
        rotation_transform.SetRotation(radians(rotation[0]), radians(rotation[1]), radians(rotation[2]))

        translation_back = itk.TranslationTransform[itk.D, 3].New()
        translation_back.SetOffset(-1.0 * np.array(rotation_center))

        # 创建组合变换
        transform = itk.CompositeTransform[itk.D, 3].New()
        transform.AddTransform(translation_to_origin)
        transform.AddTransform(rotation_transform)
        transform.AddTransform(translation_back)  

        # 应用变换
        resampler = itk.ResampleImageFilter.New(Input=itk_img)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        resampler.SetSize(itk_img.GetLargestPossibleRegion().GetSize())
        resampler.SetOutputOrigin(itk_img.GetOrigin())
        resampler.SetOutputSpacing(itk_img.GetSpacing())
        resampler.SetOutputDirection(itk_img.GetDirection())
        resampler.Update()

        # 获取重采样后的图像
        resampled_image = resampler.GetOutput()
        # 定义所需切片的大小和位置，z代表切多厚
        slice_size = [size[0], size[1], 1]
        # itk框架要定义一个region来处理slice
        region = itk.ImageRegion[3]()
        region.SetSize(slice_size)
        region.SetIndex([slice_indeces[0], slice_indeces[1], slice_indeces[2]])  # 选择开始的索引，可以调整
        # 使用 ExtractImageFilter 提取切片
        extract_filter = itk.ExtractImageFilter.New(Input=resampled_image, ExtractionRegion=region)
        extract_filter.Update()
        # 获取切片
        slice_image = extract_filter.GetOutput()
        # 将切片转换为 NumPy 数组并显示
        slice_array = itk.GetArrayFromImage(slice_image)
        return slice_array

    def mutual_information(image1, image2, bins=256):

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

    def normalized_mutual_information(image1, image2, bins=256):
        """计算归一化互信息（NMI）。"""
        hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)

        # 计算边缘熵
        h1_entropy = Tools.caculate_entropy(image1)
        h2_entropy = Tools.caculate_entropy(image2)

        # 计算联合熵
        h12_entropy = Tools.caculate_hist_entropy(hist_2d)

        # 计算 NMI
        nmi = (h1_entropy + h2_entropy) / h12_entropy
        return nmi
    
    # 这个值越大越好
    def spatial_correlation(img1, img2, threshold, bound):
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
    def spatial_correlation_with_mask(masked_img, img1, img2, bound):
        lower_bound = bound[0]
        upper_bound = bound[1]

        img1_after_masked = img1 & masked_img
        img2_after_masked = img2 & masked_img

        mask_white_bool = masked_img == 255
        mask_num = mask_white_bool.sum()

        img1_after_masked = np.int16(img1_after_masked)
        img2_after_masked = np.int16(img2_after_masked)

        diff_imgs = np.abs(img1_after_masked - img2_after_masked)
        diff_imgs = np.uint8(diff_imgs)

        # 这个下界不能包含了，不然就有问题了，因为上述位运算会出来许多的0
        above_lower_bound = diff_imgs > lower_bound
        less_upper_bound = diff_imgs <= upper_bound
        above_upper_bound = diff_imgs > upper_bound

        count = np.sum(above_lower_bound & less_upper_bound)
        penalty = np.sum(above_upper_bound)
        return (count - penalty).item() / mask_num, diff_imgs, img1_after_masked, img2_after_masked

    # 计算DICE分数，用来比较二值化的图像
    def dice_coefficient(binary_image1, binary_image2):
        # 确保输入是二值图像
        binary_image1 = np.asarray(binary_image1).astype(np.bool_)
        binary_image2 = np.asarray(binary_image2).astype(np.bool_)
        
        # 计算交集和各自的元素数
        intersection = np.logical_and(binary_image1, binary_image2)
        sum1 = binary_image1.sum()
        sum2 = binary_image2.sum()
        
        # 计算Dice系数
        dice = 2. * intersection.sum() / (sum1 + sum2)
        return dice
    
    def jaccard_index(binary_image1, binary_image2):
        binary_image1 = np.asarray(binary_image1).astype(np.bool_)
        binary_image2 = np.asarray(binary_image2).astype(np.bool_)
        intersection = np.logical_and(binary_image1, binary_image2)
        union = np.logical_or(binary_image1, binary_image2)
        return intersection.sum() / union.sum()

    def num_many_occurence_times(arr):
        ele, count = Tools.find_ith_frequent_element(arr, 1)
        return ele

    def find_ith_frequent_element(arr, i):
        # 获取元素及其出现次数
        elements, counts = np.unique(arr, return_counts=True)

        # 获取按频率排序的索引（从高到低）
        sorted_indices = np.argsort(-counts)

        # 检查i是否在合理范围内
        if i <= 0 or i > len(counts):
            raise ValueError("i is out of the valid range")

        # 获取第i多的元素，i-1因为索引是从0开始
        ith_element = elements[sorted_indices[i-1]]
        ith_count = counts[sorted_indices[i-1]]

        return ith_element, ith_count

    # 去寻找除了背景以外的区域, stats的shape为[n, 5]，代表有n-1个联通区域
    def max_index_besides_bg(stats):
        # 排除第一行
        subset_arr = stats[1:, 4]

        if subset_arr is None or subset_arr.size == 0:
            return None

        # 找到第 5 列（索引 4）的最大元素的索引
        max_index = np.argmax(subset_arr)
        # 由于我们排除了第一行，索引需要加 1
        adjusted_index = max_index + 1
        return stats[adjusted_index]

    def find_neigbor_pos(fixed_pos, stats):
        x, y = fixed_pos[0], fixed_pos[1]
        # 仍然是排除第一行，因为它是背景元素
        coordinates = stats[1:, 0:2]
        # 计算每个坐标点与 (x, y) 的欧氏距离
        distances = np.sqrt((coordinates[:, 0] - x)**2 + (coordinates[:, 1] - y)**2)
        # 找到最小距离的索引
        closest_index = np.argmin(distances)
        return closest_index+1

    def draw_region(img, num_labels, stats, centroids):
        # 创建彩色输出图像
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 绘制每个连通组件的边界框和质心
        for i in range(1, num_labels):  # 从1开始，跳过背景
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(output, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        return output

    def big_particle_penalty(stats_r, m_img, ratio_threshold = 2):
        # 计算连通区域的大小
        num_labels_m, labels_m, stats_m, centroids_m = cv2.connectedComponentsWithStats(m_img, 4, cv2.CV_32S)

        # 背景的标签是0，因此在浮动图像中选择数量最多的那个标签
        max_particle_m = Tools.max_index_besides_bg(stats_m)
        if max_particle_m is None or max_particle_m.size == 0: return 0
        max_m_x, max_m_y, particle_num_m = max_particle_m[0], max_particle_m[1], max_particle_m[4]

        max_particle_r = Tools.max_index_besides_bg(stats_r)
        max_r_x, max_r_y, max_particle_num_r = max_particle_r[0], max_particle_r[1], max_particle_r[4]

        # 找到离其最近的那几个质心区域
        closest_pos_r_index = Tools.find_neigbor_pos([max_m_x, max_m_y] ,stats_r)
        closest_particle_r = stats_r[closest_pos_r_index]

        # 比较两者的大小，如果差别太大就属于是高攀了
        particle_num_r = closest_particle_r[-1]

        ratio_m_r = particle_num_m / particle_num_r

        # 以防已经找到大的颗粒，只是没有对上的情况
        ratio_big_m_r = particle_num_m / max_particle_num_r

        if ratio_big_m_r <= ratio_threshold:
            return 0

        if ratio_m_r < ratio_threshold :
            return 0
        return ratio_m_r
    
    # 计算二值图像中连通区域的最大和最小区域, bin_img是np数组
    def count_max_and_min_connected_area(bin_img):
        min_area = 10000000000
        max_area = -1

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 遍历每个联通区域
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area : max_area = area
            if area < min_area : min_area = area
        
        return max_area, min_area

    # 将二值连通图像进行筛选，将颗粒大小维持在一个范围内，其它则填充背景色
    def filter_connected_bin_img(bin_img, min_area, max_area):
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历每个联通区域
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                # 如果面积不在指定范围内，将该区域填充为背景色
                cv2.drawContours(bin_img, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

        return bin_img
