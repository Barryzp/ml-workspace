import math, random
from scipy.stats import qmc

import numpy as np
from utils.tools import Tools

# 粒子类也抽象一下，先不抽象，之后再说
class Particle:
    def __init__(self, x0, pso_optim, id):
        self.id = id
        self.position = x0
        self.pso_optim = pso_optim

        self.velocity = np.random.random(x0.shape)
        self.best_position = np.copy(x0)
        
        self.debug = False
        fit_res = pso_optim.fitness(x0)
        self.best_value = fit_res[0]
        self.current_fitness = self.best_value

    def update_velocity(self, global_best_position):
        r1 = random.random()
        r2 = random.random()
        individual_w = self.pso_optim.individual_w
        global_w = self.pso_optim.global_w
        weight_inertia = self.pso_optim.weight_inertia

        cog_delta = (self.best_position - self.position)
        cognitive_velocity = individual_w * r1 * cog_delta


        soc_delta = (global_best_position - self.position)
        social_velocity = global_w * r2 * soc_delta
        
        self.velocity = weight_inertia * self.velocity + cognitive_velocity + social_velocity
        self.velocity = self.pso_optim.constrain_velocity(self.velocity)

    def move(self):
        self.position += self.velocity
        # constrain the position in a range
        self.position = self.pso_optim.constrain(self.position)


        fit_res = self.pso_optim.fitness(self.position)
        value = fit_res[0]

        if value > self.best_value:
            self.best_position = self.position.clone()
            self.best_value = value

class OptimBase:

    def __init__(self, config, global_share_datas = None, matched_3dct_id = None) -> None:
        # 匹配过程中的ct切片索引
        self.ct_matching_slice_index = None
        # 用于匹配过程中的全局数据共享
        self.global_share_obj = global_share_datas

        self.current_iterations = 0
        self.config = config
        self.records = []
        # 随机生成的粒子的初始值
        self.particle_vals = []

        # 最优解
        self.best_solution = None
        self.best_value = -10000
        self.best_result_per_iter = None

        # 配准的相关结果
        self.best_match_img = None
        # 设置初始位移以及旋转角度
        self.init_translate = self.config.init_translate
        self.translate_delta = self.config.translate_delta
        
        self.init_rotation = self.config.init_rotation
        self.rotation_delta = self.config.rotation_delta
        self.rot_z_delta = self.config.rot_z_delta

        # 设置匹配过程对应3d块的唯一索引
        if self.config.mode == "matched":
            self.matched_3dct_id = matched_3dct_id

    def clear_datas(self):
        self.records = []

    def set_best(self, value, solution):
        if value > self.best_value:
            self.best_value = value
            self.best_solution = solution

    # 全局的数据
    def put_best_data_in_share(self, ct3d_index, position, fitness, best_slice):
        if self.config.mode != "matched": return
        if self.global_share_obj == None: return

        pos_np = position.numpy()
        pos_np = np.insert(pos_np, 0, best_slice)
        pos_np = np.insert(pos_np, 0, ct3d_index)
        pos_np = np.insert(pos_np, pos_np.size, fitness)
        self.global_share_obj.put_in_share_objects(pos_np.tolist())

    def set_init_params(self, reg_similarity, reg_obj):
        self.init_basic_params()
        self.reg_similarity = reg_similarity
        if self.config.mode == "matched" : self.init_match_params(reg_obj)
    
    # 对于不同的ct块还不一样
    def init_match_params(self, reg_obj):
        ct_index_array = reg_obj.get_3dct_index_array(self.matched_3dct_id)
        ct_depth = len(ct_index_array)
        # Z轴上也进行了下采样，向上取整以防超出区域，这个角度其实也可以用原始的尺寸算，但是难得取，就这样也可以
        downsamp_times = self.config.downsample_times
        latent_depth = math.ceil(self.config.latent_bse_depth / downsamp_times)
        bse_height_cur, bse_width_cur = reg_obj.get_bse_img_shape()
        ct_height_cur, ct_width_cur = self.config.cropped_ct_size[0], self.config.cropped_ct_size[1]
        
        bse_height_ori, bse_width_ori = reg_obj.get_bse_ori_img_shape()
        
        # 限制x的移动范围
        # 起始的左上角坐标
        # bse切面外接圆半径
        r_slice_circle = math.sqrt(bse_width_cur**2 + bse_height_cur**2) * 0.5
        max_width_delta = math.ceil(r_slice_circle - bse_width_cur * 0.5)
        max_height_delta = math.ceil(r_slice_circle - bse_height_cur * 0.5)
        self.init_translate = [max_width_delta, max_height_delta, latent_depth]
        
        translate_delta_x = ct_width_cur - bse_width_cur - 2 * max_width_delta
        translate_delta_y = ct_height_cur - bse_height_cur - 2 * max_height_delta
        translate_delta_z = ct_depth - 2 * latent_depth
        self.translate_delta = [translate_delta_x, translate_delta_y, translate_delta_z]

        sin_theta_width = latent_depth / (bse_width_ori * 0.5)
        sin_theta_height = latent_depth / (bse_height_ori * 0.5)
        radians_height = math.asin(sin_theta_height)
        degree_height = math.degrees(radians_height)
        radians_width = math.asin(sin_theta_width)
        degree_width = math.degrees(radians_width)

        self.init_rotation = [-degree_width, -degree_height, self.init_rotation[-1]]
        rotation_delta_x = degree_width * 2
        rotation_delta_y = degree_height * 2
        rotation_delta_z = self.rot_z_delta
        self.rotation_delta = [rotation_delta_x, 
                                      rotation_delta_y, 
                                      rotation_delta_z]

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
        self.minV = np.array([
                self.init_translate[0],
                self.init_translate[1],
                init_rotation,
        ])
        self.maxV = np.array([
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

        # 这个config就不能是公用的了
        # 这个坐标需要注意，是需要记录的
        init_translate = self.init_translate
        init_rotation = self.init_rotation
        # 位移限制的范围
        translate_delta = self.translate_delta
        rotation_delta = self.rotation_delta
        # 生成初始参数规定范围，
        self.minV = np.array([
            init_translate[0], 
                init_translate[1],
                init_translate[2],
                init_rotation[0], 
                init_rotation[1],
                init_rotation_z,
        ])
        self.maxV = np.array([
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
        if self.config.mode == "matched":
            return self.reg_similarity(position, self.matched_3dct_id)
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
                       "slice_index", 
                       "translate_x", "translate_y", "translate_z", 
                       "rotation_x", "rotation_y", "rotation_z", 
                       "fitness"]
            file_name = f"{self.matched_3dct_id}_pso_params_3d_ct.csv"

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
        slice_index = fit_res[2]
        weighted_sp = fit_res[-1]
        mi_value = fit_res[-3]
        sp = fit_res[-2]

        data_item = self.best_solution.numpy()
        data_item = np.insert(data_item, 0, slice_index)
        data_item = np.insert(data_item, 0, iter)
        data_item = np.insert(data_item, data_item.size, global_best_val)
        self.records.append(data_item.tolist())

        if self.config.mode != "matched": self.save_iteration_best_reg_img(__, iter)

    # 用于判断匹配是否完成，从而及时跳出循环
    def check_match_finished(self):
        if self.config.mode != "matched": return False
        return self.global_share_obj.get_loop_state()

    # best_val为正值
    def set_global_best_datas(self, best_val, best_position, best_img, ct_matching_slice_index, volume_index):
        if self.global_share_obj == None: return
        self.global_share_obj.set_best(best_val, best_position, best_img, ct_matching_slice_index, volume_index)

    # 在匹配过程中不太一样，我们将角度化成若干份，再进行优化，主要是减少搜索空间
    def run_matched(self, total_runtimes):
        start_rot_z = self.init_rotation[-1]
        # 分成若干段
        rot_z_delta = self.rot_z_delta
        loop_times = int(360 // rot_z_delta)

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


    # 使用此方法粒子数的数量必须是2^n
    def spawn_uniform_particles(self):
        """
        使用Sobol序列生成在给定边界内均匀分布的点
        :param num_points: 点的数量
        :param bounds: 每个维度的边界，格式为 [(min_x, max_x), (min_y, max_y), (min_z, max_z)]
        :return: numpy array of points
        """
        num_points = self.particle_num
        dimension = self.parameters_num
        bounds = np.array((self.minV, self.maxV)).transpose()

        sampler = qmc.Sobol(d=dimension, scramble=True)
        sample = sampler.random_base2(m=int(np.log2(num_points)))
        scaled_sample = qmc.scale(sample, [b[0] for b in bounds], [b[1] for b in bounds])

        return np.array(scaled_sample)
    
    # 随机生成粒子数量
    def spawn_random_particles(self):
        self.particle_vals = [np.random.uniform(self.minV, self.maxV) for i in range(self.particle_num)]

    def spawn_random_particles_for_test_optim(self):
        lower_bound = self.config.solution_bound_min
        upper_bound = self.config.solution_bound_max
        d = self.config.solution_dimension
        self.particle_vals = [np.random.uniform(lower_bound, upper_bound, d) for i in range(self.particle_num)]

    # 具体算法，返回最优值
    def _algorithm(self):
        pass

    # 生成随机粒子
    def _spawn_particles(self):
        if self.config.mode == "test":
            self.spawn_random_particles_for_test_optim()
        else:
            self.spawn_random_particles()

    # 运行标准优化测试函数
    def run_std_optim(self):
        # 随机生成粒子
        self._spawn_particles()
        # 运行算法
        self._algorithm()
        return self.best_value, self.best_solution