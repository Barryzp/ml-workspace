from PIL import Image
import numpy as np

# 先预定义范围，超出这个编号的范围CT图像有点问题
AVALIABLE_RANGE = {
    1:[125, 970],
    2:[126, 1121],
    3:[125, 1162],
    4:[125, 1034],
    5:[125, 1170],
    6:[125, 1130],
    7:[125, 1200],
    8:[125, 1050],
    9:[125, 1150],
    10:[125, 1000],
    11:[125, 1120],
    12:[125, 1194],
    13:[125, 1114],

    # JPG格式的
    14:[185, 1030],
    15:[185, 1180],
    16:[185, 1345],
    17:[185, 1121],

    #BMP
    18:[185, 982],
    19:[185, 918],
    20:[185, 1030],
    21:[185, 1330],
    22:[185, 1345],
    23:[185, 1320],
    24:[185, 1110],
    25:[185, 1300],
    26:[185, 1054],
    27:[185, 1326],
    28:[185, 1326],
    29:[185, 1200],
    30:[185, 1200],
    31:[185, 1345],
    32:[185, 1200],
    33:[185, 1345],

    #TIF格式
    34:[185, 1158],
    35:[185, 1280],
    36:[185, 1200],
    37:[185, 1200],
    38:[185, 1345],
    39:[185, 1300],
    40:[185, 1250],
    41:[185, 1180],
    42:[185, 1200],
    43:[185, 1260],
    44:[185, 1260],
    45:[185, 839],
    46:[185, 1300],
    47:[185, 1250],
    48:[185, 1345],
    49:[185, 1345],
    50:[185, 1200],
    51:[185, 1345],
    52:[185, 1345],
    53:[185, 1250],
}

# 读取图像并增强后保存到文件夹
SAMPLE_NUM = 1000
SAMPLES = 53

ROOT = 'H:/CT'
BMP = ".bmp"
JPG = ".jpg"
TIF = ".tif"

def fill_digit(num, len=8):
    return str(num).zfill(len)

class CommonConfig:
    
    def set_crop_imgs_range(start_id, range):
        pass

    def get_range(sample_id):
        return AVALIABLE_RANGE[sample_id]
    
    def get_file_ex_name(sample_id):
        file_extend_name = BMP
        if sample_id >= 14:
            file_extend_name = JPG
        if sample_id >= 18:
            file_extend_name = BMP
        if sample_id >= 34:
            file_extend_name = TIF
        
        return file_extend_name

    # 获取某张id的图像
    def get_cement_ct_slice(sample_id, slice_index):
        path = f"{ROOT}/{sample_id}/Reconstruction/"
        file_extend_name = CommonConfig.get_file_ex_name(sample_id)
        ct_prefix = f"{sample_id}-_IR_rec"
        if sample_id == 6:
            ct_prefix = f"{sample_id}-_rec"
        elif sample_id == 45 or sample_id == 52:
            ct_prefix = f"{sample_id}-1-_IR_rec"
        ct_name = f"{ct_prefix}{fill_digit(slice_index)}{file_extend_name}"
        ct_path = path + ct_name

        image = None
        # 进行复制图片操作
        if file_extend_name == BMP:
            image = Image.open(ct_path).convert('L')  # 确保图像是灰度的
        else:
            # 打开并保存图像为BMP格式
            with Image.open(ct_path) as img:
                # 检查图像模式并进行转换（如果需要）
                if img.mode == 'I;16':
                    img = img.convert('I')  # 转换为32位整型灰度图像
                    img = img.point(lambda i: i * (1/256))  # 缩放到8位灰度
                    img = img.convert('L')  # 转换为8位灰度图像
                    image = img
                else:
                    image = Image.open(ct_path).convert('L')

        return np.array(image)