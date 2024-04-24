import torch, math
import numpy as np
from utils.tools import Tools

class OptimBase:

    def __init__(self, config, share_records_out = None) -> None:
        # 匹配过程中的ct切片索引
        self.ct_matching_slice_index = None
        self.share_records_out = share_records_out
        self.current_iterations = 0
        self.config = config

        # 最优解
        self.best_solution = None
        self.best_value = None
        self.best_match_img = None
        self.best_result_per_iter = None

    def put_best_data_in_share(self, fit_res, position):
        if self.config.mode != "matched": return
        if self.share_records_out == None: return
        max_val, _, mi, sp, weighted_sp = fit_res
        pos_np = position.numpy()
        pos_np = np.insert(pos_np, 0, self.ct_matching_slice_index)
        pos_np = np.insert(pos_np, pos_np.size, max_val)
        pos_np = np.insert(pos_np, pos_np.size, mi)
        pos_np = np.insert(pos_np, pos_np.size, sp)
        pos_np = np.insert(pos_np, pos_np.size, weighted_sp)
        self.share_records_out.append(pos_np.tolist())

    def set_init_params(self, refer_img_size, reg_similarity, ct_matching_slice_index = None):
        self.init_basic_params()
        self.reg_similarity = reg_similarity
        if self.config.mode == "2d":
            self.init_with_2d_params()
        elif self.config.mode == "3d":
            width, height = refer_img_size
            if height > width:
                border = width
            else:
                border = height
            self.init_with_3d_params(border)
        elif self.config.mode == "matched":
            self.ct_matching_slice_index = ct_matching_slice_index
            self.init_with_2d_params()

    # 基本参数的初始化
    def init_basic_params(self):
        self.particle_num = self.config.particle_num
        self.iteratons = self.config.iteratons

    # 2d/2d图像的寻找最优参数
    def init_with_2d_params(self):
        self.parameters_num = 3

        init_translate = self.config.init_translate
        init_rotation = 0.0

        # 位移限制的范围
        translate_delta = self.config.translate_delta
        rotation_delta = self.config.rotation_delta[-1]

        # 生成初始参数规定范围，
        self.minV = [
                init_translate[0],
                init_translate[1],
                init_rotation,
        ]
        self.maxV = [
            init_translate[0] + translate_delta[0], 
                init_translate[1] + translate_delta[1],
                init_rotation + rotation_delta, 
        ]

        # 每个粒子的移动速度是不同的, [speed_x, speed_y, speed_z, rotation_x, rotation_y, rotation_z]
        speed_x = translate_delta[0] * self.speed_param_ratio
        speed_y = translate_delta[1] * self.speed_param_ratio
        speed_rotation = rotation_delta * self.speed_param_ratio
        self.speed = torch.tensor([speed_x, speed_y, speed_rotation]) # 粒子移动的速度为参数范围的10%~20%

    # 2d/3d图像的最优参数查找
    def init_with_3d_params(self, img_border_len):
        ############## 初始参数变量范围设置，六个（translate_x,y,z, rotation_x,y,z）###################
        self.slice_num = self.config.slice_num
        self.parameters_num = 6
        
        # 这个坐标需要注意，是需要记录的
        init_translate = (self.config.init_translate[0], self.config.init_translate[1], self.slice_num * 0.5)
        init_rotation = (0.0, 0.0, 0.0)
        # 位移限制的范围
        translate_delta = self.config.translate_delta
        rotation_delta = math.degrees(math.atan((self.slice_num * 0.5)/(img_border_len * 0.5))) 
        self.rotation_center_xy = self.config.rotation_center_xy
        # 生成初始参数规定范围，
        self.minV = [
            init_translate[0], 
                init_translate[1],
                init_translate[2],
                init_rotation[0], 
                init_rotation[1],
                init_rotation[2],
        ]
        self.maxV = [
            init_translate[0] + translate_delta[0], 
                init_translate[1] + translate_delta[1],
                init_translate[2] + translate_delta[2],
                init_rotation[0] + rotation_delta, 
                init_rotation[1] + rotation_delta,
                init_rotation[2] + rotation_delta,
        ]

        # 每个粒子的移动速度是不同的, [speed_x, speed_y, speed_z, rotation_x, rotation_y, rotation_z]
        speed_x = translate_delta[0] * 2 * self.speed_param_ratio
        speed_y = translate_delta[1] * 2 * self.speed_param_ratio
        speed_z = translate_delta[0] * 2 * self.speed_param_ratio
        speed_rotation = rotation_delta * 2 * self.speed_param_ratio
        self.speed = torch.tensor([speed_x, speed_y, speed_z, speed_rotation, speed_rotation, speed_rotation]) # 粒子移动的速度为参数范围的10%~20%

    def auto_nonlinear_sp_lambda(self, k=12, a=0.7):
        all_iteration = self.config.iteratons + 1
        x = self.current_iterations / all_iteration
        return 1 / (1 + np.exp(-k * (x - a)))

    def fitness(self, position):
        sp_lambda = 1
        if self.config.auto_lambda:
            sp_lambda = self.auto_nonlinear_sp_lambda()
        return self.reg_similarity(position, self.ct_matching_slice_index, sp_lambda)

    # 保存迭代过程中的参数
    def save_iteration_params(self, records):
        file_path = Tools.get_save_path(self.config)
        file_name = f"{self.config.repeat_count}_pso_params_{self.config.mode}.csv"

        if self.config.mode == "2d" :
            columns = ["iterations", "x", "y", "rotation", "fitness", "weighted_sp", "mi", "sp"]
        elif self.config.mode == "3d":
            columns = ["iterations", 
                       "x", "y", "z", 
                       "rotation_x", "rotation_y", "rotation_z",
                       "fitness"]
        elif self.config.mode == "matched":
            columns = ["iterations", "x", "y", "rotation", "fitness"]
            file_name = f"pso_params_{self.ct_matching_slice_index}.csv"

        Tools.save_params2df(records, columns, file_path, file_name)

    def save_iteration_best_reg_img(self, img_array, iterations):
        file_path = Tools.get_save_path(self.config)
        file_name = f"iter{iterations}_best_reg.bmp"
        if self.config.mode == "matched": file_name = f"best_reg_{self.ct_matching_slice_index}.bmp"
        Tools.save_img(file_path, file_name, img_array)

    def save_iter_records(self, records, iter):
        if records == None: return

        fit_res = self.best_result_per_iter
        global_best_val = fit_res[0]
        __ = fit_res[1]
        weighted_sp = fit_res[-1]
        mi_value = fit_res[-3]
        sp = fit_res[-2]

        data_item = self.best_solution.numpy()
        data_item = np.insert(data_item, 0, iter)
        data_item = np.insert(data_item, data_item.size, global_best_val)
        data_item = np.append(data_item, [weighted_sp, mi_value, sp])
        records.append(data_item.tolist())

        if self.config.mode != "matched": self.save_iteration_best_reg_img(__, iter)

    # 具体算法，返回最优值
    def _algorithm(self, particle_vals, num_iterations, record):
        pass

    # 进行优化
    def run(self):
        pass