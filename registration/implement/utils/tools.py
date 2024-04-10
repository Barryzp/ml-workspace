import cv2, os
import numpy as np
import itk, yaml
import pandas as pd
from math import radians
from munch import Munch
from pathlib import Path
from PIL import Image

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
        # 将数据保存为 YAML 文件
        with open(file_path, 'w') as file:
            yaml.dump(obj, file)

    def check_file(folder_path, file_name):
        file_path = os.path.join(folder_path, file_name)
        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            # 如果文件夹不存在，创建它
            os.makedirs(folder_path)
        else:
            # 如果文件夹存在，检查特定文件是否也存在
            if os.path.exists(file_path):
                # 如果文件存在，抛出异常
                raise FileExistsError(f"The file '{file_path}' already exists.")
            print(f"Folder '{folder_path}' already exists and the file '{file_name}' is not present.")
        
        return file_path
    
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