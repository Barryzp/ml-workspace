# 金字塔配置
class PyramidCfg:
    def __init__(self) -> None:
        # 下采样倍数
        self.downsample_times = None
        # 当前采样倍数下的采样结果
        self.reg_result_img = None
        # 当前采样倍数下的ct图像大小
        self.reg_ct_img_size = None
        # 当前采样倍数下的bse图像大小
        self.reg_bse_img_size = None
        # bse indeces，BSE图像在ct图像中对应的索引
        self.reg_bse_indeces = None
        # 起始start_idx
        self.start_idx = -1
        # end index，不包括
        self.end_idx = -1
        # 配准得到的偏移参数
        self.translation = None
        # 这个旋转角度是相对于图像本身
        self.rotation = None
        self.delta_translation = None
        self.delta_rotation = None

        # 用于配准的浮动BSE图像
        self.bse_img = None

    
    # 设置配准参数
    def set_reg_params(self, translation, rotation, reg_res):
        self.rotation = rotation
        self.translation = translation
        self.reg_result_img = reg_res

    def to_dict(self):
        return {
            "translation" : self.translation, 
            "rotation" : self.rotation,
            "downsample_times" : self.downsample_times
        }

    # 打印配准参数
    def print_reg_params(self):
        print(f"downsample_times: {self.downsample_times}")
        print(f"reg transltation: {self.translation}")
        print(f"reg rotation: {self.rotation}")

    # 构建金字塔配置
    def buildPyramid(downsample_times, bse_img_size, ct_img_size):
        pyramid = PyramidCfg()
        pyramid.downsample_times = downsample_times
        pyramid.reg_bse_img_size = bse_img_size
        pyramid.reg_ct_img_size = ct_img_size
        return pyramid