import cv2, os
import numpy as np
import itk, yaml
import pandas as pd
from scipy import stats
from math import radians
from munch import Munch
from pathlib import Path
from PIL import Image, ImageDraw
from .common_config import CommonConfig


class Tools:
    # 加载yaml文件并转化成一个对象，注意：路径是决定路径
    def load_yaml_config(path):
        # 使用 Path 对象处理文件路径
        file_path = Path(path)
        if not file_path.is_absolute():
            # 如果路径不是绝对路径，则将其转换为绝对路径
            file_path = Path.cwd() / file_path

        with open(file_path, 'r', encoding='utf-8') as f:
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

    def get_processed_bse_path(config):
        sample_index = config.cement_sample_index
        bse_sample_index = config.sample_bse_index

        zoom_times = config.bse_zoom_times
        middle = zoom_times // 100
        end_index = config.zoom_bse_index

        file_pref = f"{sample_index}-{middle}-{end_index}"
        # BSE 图像裁剪以及对比度增强处理
        src_path = f"D:/workspace/ml-workspace/registration/datasets/sample{sample_index}/bse/s{bse_sample_index}/{zoom_times}"
        return src_path, file_pref

    def get_ct_img(cement_id, slice_index):
        return CommonConfig.get_cement_ct_slice(cement_id, slice_index)

    # 保存图片
    def save_img(folder_path, file_name, img):
        # 检查一下
        file_path = Tools.check_file(folder_path, file_name)
        # 保存图像, 这个img是np数组类型的
        saved_img = Image.fromarray(img.astype(np.uint8))
        saved_img.save(file_path)

    def save_img_jpg(folder_path, file_name, img, jpg_quality):
        # 检查一下
        file_path = Tools.check_file(folder_path, file_name)
        # 保存图像, 这个img是np数组类型的
        saved_img = Image.fromarray(img.astype(np.uint8))
        saved_img.save(file_path, "JPEG", quality=jpg_quality, optimize=True, progressive=True)
    
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

    # 下采样二值图像
    def downsample_bin_img(image_np, downsample_times, size = None):
        # 使用缩放因子来减半图片尺寸
        resized_image = Tools.downsample_image(image_np, downsample_times, size)
        # 对下采样后的图像进行二值化处理
        _, binary_downsampled_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
        return binary_downsampled_image

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
    
    def most_frequent_element(arr):
        # 获取唯一元素及其出现的频率
        unique_elements, counts = np.unique(arr, return_counts=True)
        
        # 获取出现频率最高的元素的索引
        max_count_index = np.argmax(counts)
        
        # 返回出现频率最高的元素
        return unique_elements[max_count_index]

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

        # 寻找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, 4, cv2.CV_32S)

        # 遍历所有连通区域
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_area : max_area = area
            if area < min_area : min_area = area
        
        return max_area, min_area

    # 将二值连通图像进行筛选，将颗粒大小维持在一个范围内，其它则填充背景色
    def filter_connected_bin_img(bin_img, min_area, max_area):
        # 寻找连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, 4, cv2.CV_32S)

        # 遍历每个联通区域
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                # 如果面积不在指定范围内，将该区域填充为背景色
                bin_img[labels == i] = 0

        return bin_img
    
    # 计算前一个二值化后的图像与分割后的图像中，处于前面10个最大连通区域中灰度出现最高的统计，都为numpy数组
    def bin_mask_mode_gray_cls_10(bin_img, seg_img):
        # 计算连通组件及其统计信息
        num_labels, labels, stats_, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=4, ltype=cv2.CV_32S)
        # 获取连通组件的面积
        areas = stats_[:, cv2.CC_STAT_AREA]

        # 排除背景（标签0），获取前十个最大的连通组件的索引
        top_10_indices = np.argsort(areas[1:])[::-1][:10] + 1
        gray_stat = np.array([])
        # 遍历前十个最大的连通组件
        for idx in top_10_indices:
            mask = (labels == idx).astype(np.uint8)  # 创建当前连通组件的掩码
            component_pixels = seg_img[mask == 1]  # 获取连通组件内的所有灰度值
            # 计算灰度直方图
            hist = cv2.calcHist([component_pixels], [0], None, [256], [0, 256])

            # 打印一些统计信息
            mean_val = np.mean(component_pixels)
            std_val = np.std(component_pixels)
            # 计算众数
            mode_result = stats.mode(component_pixels)
            # mode_result.mode给出众数，mode_result.count给出对应的频次
            most_common_value = mode_result.mode
            frequency = mode_result.count
            gray_stat = np.append(gray_stat, most_common_value)

        # 计算众数
        mode_result = stats.mode(gray_stat)
        return int(mode_result.mode)

    # 二值化图像, gray_cls代表为白色的灰度值
    def binarized_img(image, gray_cls):
        neg_cls = image != gray_cls
        positive_cls = image == gray_cls
        image[neg_cls] = 0
        image[positive_cls] = 255
        return image
    
    # 对点进行位移，translation：(translation_x, translation_y, translation_z)
    def translate_points(translation, points):
        # 注意这个地方的point具体的坐标是[rows, cols, dim]，而对于我们的translation数组，
        # 它应该是[width_delta, height_delta, z_delta]，因此需要转换
        translation[0], translation[1] = translation[1], translation[0]

        translation = np.array(translation)
        # 需要在这个地方进行判定不，不能为负，为负就跳出异常
        after_translate = points + translation
        if after_translate.min() < 0 : raise ValueError("索引小于零，访问越界！")
        return after_translate


    def rotation_matrix_from_euler_angles(angles, order='XYZ'):
        """
        根据欧拉角生成旋转矩阵。
        参数：
        angles -- 欧拉角，长度为 3 的数组，表示绕每个轴的旋转角度
        order -- 旋转顺序，默认为 'XYZ'
        返回：
        rotation_matrix -- 旋转矩阵，形状为 (3, 3)
        """
        theta_x, theta_y, theta_z = angles

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        Ry = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        Rz = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        if order == 'XYZ':
            return np.dot(Rz, np.dot(Ry, Rx))
        elif order == 'ZYX':
            return np.dot(Rx, np.dot(Ry, Rz))
        else:
            raise ValueError("Unsupported rotation order")

    # rotation_center：就是切片的中心对应的那个点
    # points: 切片
    # rotations: (rotation_x, rotation_y, rotation_z)都是角度
    def rotate_points(rotation_center, rotations, points):
        # 定义旋转角度（弧度制）
        theta_x = np.radians(rotations[0])  # 绕 x 轴旋转角度
        theta_y = np.radians(rotations[1])  # 绕 y 轴旋转角度
        theta_z = np.radians(rotations[2])  # 绕 z 轴旋转角度

        # 根据欧拉角生成旋转矩阵，固定旋转顺序为 'XYZ'
        rotation_matrix = Tools.rotation_matrix_from_euler_angles([theta_x, theta_y, theta_z], order='XYZ')

        translate_2_origin = np.array(rotation_center)

        points_in_origin = points - translate_2_origin
        points_after_rotate = np.dot(points_in_origin, rotation_matrix)

        points_after_transform = points_after_rotate + translate_2_origin

        # 注意这个地方跑出了边界外的处理，需要进行限制，两个方面限制
        # （1）确实就是超出去了怎么办，其中有负值，那么怎么处理，这个地方设置不当容易造成溢出
        # （2）可以事先进行先验的范围设置
        # 这个地方不能直接这样暴力地转换，要进行四舍五入的处理
        min_value = points_after_transform.min()
        if min_value < 0 :
            arr = points_after_transform
            min_index = np.argmin(arr)
            min_index_multi_dim = np.unravel_index(min_index, arr.shape)
            index = arr[min_index_multi_dim[:2]]
            raise ValueError("索引小于零，访问越界！")

        return points_after_transform

    def force_convert_uint(index_array):
        index_array = np.round(index_array)
        index_array = index_array.astype(np.uint16)
        return index_array

    # 三次线性插值
    def trilinear_interpolation(ct_image, index_array):
        """
        手动实现三线性插值。
        
        参数:
        ct_image -- 三维CT图像，形状为 (depth, height, width)
        index_array -- 插值点的索引数组，形状为 (width, height, 3)
        
        返回:
        values -- 插值后的值，形状为 (width, height)
        """
        def get_value_at_point(ct_image, z, y, x):
            depth, height, width = ct_image.shape
            
            # 获取八个顶点的坐标
            x0, x1 = int(np.floor(x)), int(np.ceil(x))
            y0, y1 = int(np.floor(y)), int(np.ceil(y))
            z0, z1 = int(np.floor(z)), int(np.ceil(z))
            
            # 确保坐标不越界
            x0, x1 = max(0, x0), min(width - 1, x1)
            y0, y1 = max(0, y0), min(height - 1, y1)
            z0, z1 = max(0, z0), min(depth - 1, z1)
            
            # 获取八个顶点的值
            c000 = ct_image[z0, y0, x0]
            c001 = ct_image[z0, y0, x1]
            c010 = ct_image[z0, y1, x0]
            c011 = ct_image[z0, y1, x1]
            c100 = ct_image[z1, y0, x0]
            c101 = ct_image[z1, y0, x1]
            c110 = ct_image[z1, y1, x0]
            c111 = ct_image[z1, y1, x1]
            
            # 计算插值系数
            xd = (x - x0) / (x1 - x0) if x1 != x0 else 0
            yd = (y - y0) / (y1 - y0) if y1 != y0 else 0
            zd = (z - z0) / (z1 - z0) if z1 != z0 else 0
            
            # 线性插值
            c00 = c000 * (1 - xd) + c001 * xd
            c01 = c010 * (1 - xd) + c011 * xd
            c10 = c100 * (1 - xd) + c101 * xd
            c11 = c110 * (1 - xd) + c111 * xd
            
            c0 = c00 * (1 - yd) + c01 * yd
            c1 = c10 * (1 - yd) + c11 * yd
            
            c = c0 * (1 - zd) + c1 * zd
            
            return c
        
        width, height = index_array.shape[:2]
        values = np.zeros((width, height))
        
        for i in range(width):
            for j in range(height):
                y, x, z = index_array[i, j]
                values[i, j] = get_value_at_point(ct_image, z, y, x)
        
        return np.clip(values, 0, 255)  # 确保值在0-255范围内