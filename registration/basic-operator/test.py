import itk
import numpy as np
from glob import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

BMP = ".bmp"
JPG = ".jpg"
TIF = ".tif"
# 指定图像序列的目录
ROOT = "G:/CT"

sample_id = 6
start_id = 185
end_id = 200

image_path = f"{ROOT}/{sample_id}/Reconstruction/"
if sample_id >= 1:
    file_extend_name = BMP
if sample_id >= 14:
    file_extend_name = JPG
if sample_id >= 18:
    file_extend_name = BMP
if sample_id >= 34:
    file_extend_name = TIF

def get_path(index):
    ct_prefix = f"{sample_id}-_IR_rec"
    if sample_id == 6:
        ct_prefix = f"{sample_id}-_rec"
    elif sample_id == 45 or sample_id == 52:
        ct_prefix = f"{sample_id}-1-_IR_rec"
    ct_name = f"{ct_prefix}{str(index).zfill(8)}{BMP}"
    ct_path = f"{image_path}/{ct_name}"
    return ct_path

arr_len = end_id - start_id
image_filenames = [get_path(i+start_id) for i in range(arr_len)]

import itk
import numpy as np
import matplotlib.pyplot as plt

image = itk.imread("D:/workspace/ml-workspace/registration/basic-operator/test.nrrd")


# 定义变换（以沿任意方向切片为例）
transform = itk.AffineTransform[itk.D, 3].New()
rotation_axis = (1, 0, 0)  # 以 x 轴为例，可根据需要调整
rotation_angle = np.pi / 4  # 旋转45度，可根据需要调整
transform.Rotate3D(rotation_axis, rotation_angle, False)

# 设置重采样过滤器
resampler = itk.ResampleImageFilter.New(Input=image)
resampler.SetTransform(transform)
resampler.SetSize(image.GetLargestPossibleRegion().GetSize())
resampler.SetOutputOrigin(image.GetOrigin())
resampler.SetOutputSpacing(image.GetSpacing())
resampler.SetOutputDirection(image.GetDirection())
resampler.Update()

# 获取重采样后的图像
resampled_image = resampler.GetOutput()

# 可视化一个切片
slice_index = 50  # 示例索引，可根据需要调整
slice_array = itk.array_view_from_image(resampled_image)[slice_index, :, :]
plt.imshow(slice_array, cmap='gray')
plt.axis('off')
plt.show()