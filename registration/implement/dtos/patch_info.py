# 金字塔配置
class PyramidCfg:
    def __init__(self) -> None:
        # 下采样
        self.downsample_times = None
        self.r_img = None
        # 这个translate是相对于bse图像左上角的点
        self.delta_translate_bse = None
        # 这个translate是相对于ct左上角的点
        self.delta_translate_ct = None
        # 这个旋转角度是相对于图像本身
        self.delta_rotation = None
        # 当前这个patch的大小[width, height]
        self.patch_size = None
    
    def buildPyramid(downsample_times, r_img, translate_ct, translate_bse, rotation):
        pyramid = PyramidCfg()
        pyramid.downsample_times = downsample_times
        pyramid.r_img = r_img
        if r_img != None: pyramid.patch_size = [r_img.shape[1], r_img.shape[0]]
        pyramid.delta_translate_bse = translate_bse
        pyramid.delta_translate_ct = translate_ct
        pyramid.delta_rotation = rotation
        return pyramid

# Patch配置
class PatchCfg:
    def __init__(self) -> None:
        self.id = None
        self.z_slice = None
        self.pyramids = None
    
    def set_info(self, id, z_slice, pyramids):
        self.id = id
        self.z_slice = z_slice
        self.pyramids = pyramids

# 示例使用
# product = ProductDTO(1, 'Laptop', 999.99, 10)
# print(product)
