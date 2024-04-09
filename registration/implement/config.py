# 配置，这个配置还是得分个类，不然难得管理，明天干吧 HACK
class Config:

    record_id = "test-1"
    data_save_path = "D:/workspace/ml-workspace/registration/result"

    # 直方图划分
    bins = 256
    # spatial_information阈值设定，ct对应0项，bse对应1项，大于这个阈值的才保留
    threshold = [128, 180]
    # spatial_information差异值在多大范围内才进行统计
    bound = [0, 50]
    # spatial_information占的权重系数
    lamda_mis = 1

    # ct图像开始索引
    start_index = 10
    # ct图像终止索引
    end_index = 49
    # 图像路径
    data_path = "D:/workspace/ml-workspace/registration/datasets"
    # 对应第几个水泥样本
    cement_sample_index = 4
    # bse放大倍数
    bse_zoom_times = 100
    # 对应bse放大某个倍数下，第几个BSE成像图像
    sample_bse_index = 1
    # CT图像的XOY平面上的旋转中心
    rotation_center_xy = (960.0, 960.0)
    # 最大允许的平移量范围
    translate_delta = (20.0, 15.0, 10.0)

    # 模式：2d代表2d/2d配准，也就是CT切片直接和BSE图像配准，这个主要用在对比上面
    # 3d代表2d/3d配准，真实进行的配准
    mode = "2d"

    # 2d索引
    ct_2d_index = 32

    # PSO 参数
    # 粒子个数
    particle_num = 100
    # 迭代次数
    iteratons = 10
    # HACK 可以进行改进 惯性权重，范围在[0.4, 2]，为1回退到基本粒子群算法，值越大全局寻优能力强，局部寻优能力弱；反之值越小局部寻优能力强
    weight_inertia = 0.5  # Inertia weight
    # 个体学习因子，朝着个体最佳方向
    individual_w = 1.6    # Cognitive (particle's best) weight
    # 群体学习因子，朝着全局最佳方向
    global_w = 2.6    # Social (swarm's best) weight
    # 粒子速度的变化范围的比例，一般是粒子的取值范围的10%~20%
    speed_param_ratio = 0.1 # 0.1 ~ 0.2
