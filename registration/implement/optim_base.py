import torch, math
import numpy as np
from utils.tools import Tools

class OptimBase:

    def __init__(self, config, global_share_datas = None) -> None:
        # 匹配过程中的ct切片索引
        self.ct_matching_slice_index = None
        # 用于匹配过程中的全局数据共享
        self.global_share_obj = global_share_datas

        self.current_iterations = 0
        self.config = config
        self.records = []

        # 最优解
        self.best_solution = None
        self.best_value = -10000
        self.best_match_img = None
        self.best_result_per_iter = None

        # 设置初始位移以及旋转角度
        self.init_translate = self.config.init_translate
        self.translate_delta = self.config.translate_delta
        
        self.init_rotation = self.config.init_rotation
        self.rotation_delta = self.config.rotation_delta

    def clear_datas(self):
        self.records = []

    def set_best(self, value, solution):
        if value > self.best_value:
            self.best_value = value
            self.best_solution = solution

    def put_best_data_in_share(self, fit_res, position):
        if self.config.mode != "matched": return
        if self.global_share_obj == None: return
        max_val, _, latent_z_slice, mi, sp, weighted_sp = self.best_result_per_iter
        pos_np = position.numpy()
        pos_np = np.insert(pos_np, 0, latent_z_slice)
        pos_np = np.insert(pos_np, pos_np.size, max_val)
        pos_np = np.insert(pos_np, pos_np.size, mi)
        pos_np = np.insert(pos_np, pos_np.size, sp)
        pos_np = np.insert(pos_np, pos_np.size, weighted_sp)
        self.global_share_obj.put_in_share_objects(pos_np.tolist())

    def set_init_params(self, reg_similarity):
        self.init_basic_params()
        self.reg_similarity = reg_similarity

    # 基本参数的初始化
    def init_basic_params(self):
        self.particle_num = self.config.particle_num
        self.iteratons = self.config.iteratons

    # 2d/2d图像的寻找最优参数
    def init_with_2d_params(self):
        self.parameters_num = 3

        init_rotation = self.init_rotation[-1]
        rotation_delta = self.rotation_delta[-1]

        # 生成初始参数规定范围，
        self.minV = torch.tensor([
                self.init_translate[0],
                self.init_translate[1],
                init_rotation,
        ])
        self.maxV = torch.tensor([
                self.init_translate[0] + self.translate_delta[0], 
                self.init_translate[1] + self.translate_delta[1],
                init_rotation + rotation_delta, 
        ])

    # 2d/3d图像的最优参数查找
    def init_with_3d_params(self):
        self.parameters_num = 6
        ############## 初始参数变量范围设置，六个（translate_x,y,z, rotation_x,y,z）###################
        # 参数的范围在registration中已经设置完毕了

        # z轴的旋转角度分段优化
        init_rotation_z = self.init_rotation[-1]
        rotation_delta_z = self.rotation_delta[-1]

        # 这个坐标需要注意，是需要记录的
        init_translate = self.config.init_translate
        init_rotation = self.config.init_rotation
        # 位移限制的范围
        translate_delta = self.config.translate_delta
        rotation_delta = self.config.rotation_delta
        # 生成初始参数规定范围，
        self.minV = torch.tensor([
            init_translate[0], 
                init_translate[1],
                init_translate[2],
                init_rotation[0], 
                init_rotation[1],
                init_rotation_z,
        ])
        self.maxV = torch.tensor([
            init_translate[0] + translate_delta[0], 
                init_translate[1] + translate_delta[1],
                init_translate[2] + translate_delta[2],
                init_rotation[0] + rotation_delta[0], 
                init_rotation[1] + rotation_delta[1],
                init_rotation_z + rotation_delta_z,
        ])

    def auto_nonlinear_sp_lambda(self, k=12, a=0.7):
        all_iteration = self.config.iteratons + 1
        x = self.current_iterations / all_iteration
        return 1 / (1 + np.exp(-k * (x - a)))

    def fitness(self, position):
        sp_lambda = 1
        if self.config.auto_lambda:
            sp_lambda = self.auto_nonlinear_sp_lambda()
        return self.reg_similarity(position)

    # 保存迭代过程中的参数
    def save_iteration_params(self):
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
            columns = ["iterations", 
                       "translate_x", "translate_y", "translate_z", 
                       "rotation_x", "rotation_y", "rotation_z", 
                       "fitness", "weighted_sp", "mi", "sp"]
            file_name = f"pso_params_3d_ct.csv"

        Tools.save_params2df(self.records, columns, file_path, file_name)

    def save_iteration_best_reg_img(self, img_array, iterations):
        file_path = Tools.get_save_path(self.config)
        file_name = f"iter{iterations}_best_reg.bmp"
        if self.config.mode == "matched": file_name = f"best_reg_3dct.bmp"
        Tools.save_img(file_path, file_name, img_array)

    def save_iter_records(self, iter):

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
        self.records.append(data_item.tolist())

        if self.config.mode != "matched": self.save_iteration_best_reg_img(__, iter)

    # 具体算法，返回最优值
    def _algorithm(self, particle_vals, num_iterations, record):
        pass

    # 用于判断匹配是否完成，从而及时跳出循环
    def check_match_finished(self):
        if self.config.mode != "matched": return False
        return self.global_share_obj.get_loop_state()

    # best_val为正值
    def set_global_best_datas(self, best_val, best_position, best_img, ct_matching_slice_index):
        if self.global_share_obj == None: return
        self.global_share_obj.set_best(best_val, best_position, best_img, ct_matching_slice_index)

    # 在匹配过程中不太一样，我们将角度化成若干份，再进行优化，主要是减少搜索空间
    def run_matched(self, total_runtimes):
        start_rot_z = self.init_rotation[-1]
        # 分成若干段
        rot_z_delta = self.config.rot_z_delta
        loop_times = 1 #int(360 // rot_z_delta)

        total_iterations = total_runtimes * loop_times
        g_iter = 0

        for i in range(loop_times):
            self.init_rotation[-1] = start_rot_z + rot_z_delta * i
            if self.init_rotation[-1] >= 360.0 : self.init_rotation[-1] = 0
            print(f"================================rotation: {rot_z_delta * i}=====================================\n")
            for j in range(total_runtimes):
                self.run()
                g_iter+=1
                if self.check_match_finished() : return self.global_share_obj.global_best_value, self.global_share_obj.global_best_img
                print(f"================================{(g_iter/total_iterations) * 100}%================================")

        self.save_iteration_params()
        print(f"The maximum value of the function is: {self.best_value}")
        print(f"The best position found is: {self.best_solution}")

    # 进行优化
    def run(self):
        # 将参数初始化一下
        mode = self.config.mode
        if mode == "2d":
            self.init_with_2d_params()
        elif mode == "matched":
            self.init_with_3d_params()

    # 反复进行循环run函数，目的是寻找最优
    def run_matched_with_loops(self):
        loop_times = self.config.match_loop_times
        self.run_matched(loop_times)