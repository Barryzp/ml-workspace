import math
from scipy.stats import qmc
import logging
import numpy as np
from utils.tools import Tools

# 粒子类也抽象一下，先不抽象，之后再说
class Particle:
    def __init__(self, optim, id):
        self.id = id
        self.debug = False

        dim = len(optim.minV)
        position = np.random.uniform(optim.minV, optim.maxV)
        self.position = position
        self.optim = optim

        self.velocity = np.random.uniform(-1, 1, dim)
        self.pbest_position = position.copy()
        
        fit_res = optim.fitness(position)
        self.pbest_value = self.uppack_fitness(fit_res)
        self.current_fitness = self.pbest_value

    def uppack_fitness(self, fit_res):
        if self.optim.config.mode == "test":
            return fit_res
        else: return fit_res[0]

    def update(self, global_best_position):
        dim = self.pbest_position.shape[0]
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        individual_w = self.optim.individual_w
        global_w = self.optim.global_w
        weight_inertia = self.optim.weight_inertia

        cog_delta = (self.pbest_position - self.position)
        cognitive_velocity = individual_w * r1 * cog_delta

        soc_delta = (global_best_position - self.position)
        social_velocity = global_w * r2 * soc_delta
        
        self.velocity = weight_inertia * self.velocity + cognitive_velocity + social_velocity
        self.position += self.velocity
        self.check()

    # 检查速度和position是否在范围内
    def check(self):
        self.velocity = self.optim.constrain_velocity(self.velocity)
        self.position = self.optim.constrain(self.position)

    def evaluate(self):
        fit_res = self.optim.fitness(self.position)
        value = self.uppack_fitness(fit_res)
        self.current_fitness = value

        if value > self.pbest_value:
            self.pbest_position = np.copy(self.position)
            self.pbest_value = value

class OptimBase:

    def __init__(self, config, global_share_datas = None, matched_3dct_id = None) -> None:
        # 匹配过程中的ct切片索引
        self.ct_matching_slice_index = None
        # 用于匹配过程中的全局数据共享
        self.global_share_obj = global_share_datas
        self.matched_3dct_id = -1

        self.current_iterations = 0
        self.config = config

        self.current_fes = 0
        self.max_fes = config.max_fes

        # 用于保存迭代过程中寻找到的最优fitness
        self.records = []
        # 用于保存评估过程中寻找到的最优fitness
        self.records_fes = []

        # 最优解
        self.best_solution = None
        self.best_value = -np.inf
        self.best_result_per_iter = None
        self.run_id = -1

        # 配准的相关结果
        self.best_match_img = None

        if self.config.mode != "test":
            # 设置初始位移以及旋转角度
            self.init_translate = self.config.init_translate
            self.translate_delta = self.config.translate_delta

            self.init_rotation = self.config.init_rotation
            self.rotation_delta = self.config.rotation_delta
            self.rot_z_delta = self.config.rot_z_delta

        # 设置匹配过程对应3d块的唯一索引
        if self.config.mode == "matched":
            self.matched_3dct_id = matched_3dct_id
    
    def gen_random_position(self):
        position = np.random.uniform(self.minV, self.maxV)
        return position

    # 检测当前评估次数是否到达最大
    def check_end(self):
        return self.current_fes > self.max_fes

    # 增加评估次数
    def add_fes(self):
        self.current_fes+=1

    # 获取当前评估次数
    def get_fes(self):
        return self.current_fes

    # 用来区别每一次运行保存的值
    def set_runid(self, run_id):
        self.run_id = run_id

    def clear_datas(self):
        self.records = []
        self.records_fes = []

    def set_best(self, value, solution):
        if value > self.best_value:
            self.best_value = value
            self.best_solution = np.copy(solution)

    # 全局的数据
    def put_best_data_in_share(self, ct3d_index, position, fitness, best_slice):
        if self.config.mode != "matched": return
        if self.global_share_obj == None: return

        pos_np = position
        pos_np = np.insert(pos_np, 0, best_slice)
        pos_np = np.insert(pos_np, 0, ct3d_index)
        pos_np = np.insert(pos_np, pos_np.size, fitness)
        self.global_share_obj.put_in_share_objects(pos_np.tolist())

    def set_init_params(self, fitness_fun, reg_obj, fun_id = None):
        self.init_basic_params()
        self.fun_id = fun_id

        self.fitness_fun = fitness_fun
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
        if self.config.mode == "matched":
            return self.fitness_fun(position, self.matched_3dct_id)
        return self.fitness_fun(position)

    # 保存迭代过程中的参数
    def save_iteration_params(self):
        file_path = Tools.get_save_path(self.config)
        file_name = f"pso_params_{self.config.mode}.csv"

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
            file_name = f"{self.run_id}_{self.matched_3dct_id}_pso_params_3d_ct.csv"

        Tools.save_params2df(self.records, columns, file_path, file_name)

    # 保存迭代中的适应值以及最优解
    def save_iteration_fitness_for_test_optim(self):
        if len(self.records) == 0: return

        method_name = self.__class__.__name__

        file_path = f"{Tools.get_save_path(self.config)}/{self.fun_id}"
        file_name = f"{method_name}_iter_{self.run_id}.csv"
        columns = ["iterations", "fitness"]
        
        # best solution 也需要保存一下
        solution_item = self.best_solution.tolist()
        solution_keys = [f"x_{i}" for i in range(len(solution_item))]
        solution_file_name = f"{method_name}_solution_{self.run_id}.csv"
        # Tools.save_params2df([solution_item], solution_keys, file_path, solution_file_name)

        Tools.save_params2df(self.records, columns, file_path, file_name)

    # 保存评估中的适应值以及最优解
    def save_iteration_fes_for_test_optim(self):
        if len(self.records_fes) == 0: return

        method_name = self.__class__.__name__

        file_path = f"{Tools.get_save_path(self.config)}/{self.fun_id}"
        file_name = f"{method_name}_fes_{self.run_id}.csv"
        columns = ["FEs", "fitness"]
        
        # best solution 也需要保存一下
        solution_item = self.best_solution.tolist()
        solution_keys = [f"x_{i}" for i in range(len(solution_item))]
        solution_file_name = f"{method_name}_solution_{self.run_id}.csv"
        # Tools.save_params2df([solution_item], solution_keys, file_path, solution_file_name)

        Tools.save_params2df(self.records_fes, columns, file_path, file_name)


    def save_iteration_best_reg_img(self, img_array, fes):
        file_path = Tools.get_save_path(self.config)
        file_name = f"fes_{fes}_best_reg.bmp"
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

        data_item = self.best_solution
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
        g_iter = 0
        for j in range(total_runtimes):
            self.run()
            g_iter+=1
            if self.check_match_finished() : return self.global_share_obj.global_best_value, self.global_share_obj.global_best_img
            print(f"================================{(g_iter/total_runtimes) * 100}%================================")

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

    def recording_data_item_for_std_optim(self, iterations):
        # HACK 能够绝对值化的前提在于已经平移到原点了

        # 不保存迭代次数了，没意思
        cur_iter_best = abs(self.best_value)

        if iterations % 50 == 0:
            print(f"iterations: {iterations}, fitness: {cur_iter_best}")

        data_item = [iterations, cur_iter_best]
        self.records.append(data_item)

    # 记录当前
    def recording_data_item_FEs(self):
        if self.config.mode == "matched":
            self.recording_data_item_for_reg()
        elif self.config.mode == "test":
            self.recording_data_item_for_optim_test()

    # 记录当前
    def recording_data_item_for_optim_test(self):
        eval_times = self.get_fes()
        # 每隔一段次数记录一下
        if eval_times % self.config.save_fes_interval != 0: return

        cur_iter_best = abs(self.best_value)

        if eval_times % 5000 == 0:
            print(f"{self.__class__.__name__}, eval_times: {eval_times}, fitness: {cur_iter_best}", flush=True)
            # logging.info(f"{self.__class__.__name__}, eval_times: {eval_times}, fitness: {cur_iter_best}")
        data_item = [eval_times, cur_iter_best]
        self.records_fes.append(data_item)

    # 记录哪些数据：1. 迭代次数；2.当前迭代的全局最佳粒子；2.
    def recording_data_item_for_reg(self):
        current_fes = self.get_fes()
        if current_fes % 100 != 0:
            return

        # 记录当前迭代最佳粒子,global也需要记录一下        
        iter_best_position = self.best_solution
        fit_res = self.fitness(iter_best_position)

        cur_iter_best = fit_res[0]
        __ = fit_res[1]
        data_item = iter_best_position

        if self.config.mode == "matched":
            z_index = fit_res[2]
            data_item = np.insert(data_item, 0, z_index)
            if current_fes / 10000 == 0:
                self.save_iteration_best_reg_img(__, current_fes)
            print(f"fes: {current_fes}, fitness: {cur_iter_best}, params: {iter_best_position}")
            self.set_global_best_datas(cur_iter_best, iter_best_position, __, z_index, self.matched_3dct_id)

        data_item = np.insert(data_item, 0, current_fes)
        data_item = np.insert(data_item, data_item.size, cur_iter_best)
        self.records.append(data_item.tolist())


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
        test_fun_cfg = self.config.fun_configs[self.fun_id]
        lower_bound = test_fun_cfg["min_bound"]
        upper_bound = test_fun_cfg["max_bound"]

        d = self.config.solution_dimension

        self.minV = np.full(d, lower_bound)
        self.maxV = np.full(d, upper_bound)

        self.particle_vals = [np.random.uniform(lower_bound, upper_bound, d) for i in range(self.particle_num)]

    # 设置取值的区间范围
    def _set_bound(self):
        if self.config.test_type == "normal":
            test_fun_cfg = self.config.fun_configs[self.fun_id]
        elif self.config.test_type == "cec2013":
            test_fun_cfg = self.config.fun_configs["cec2013"][self.fun_id]
        lower_bound = test_fun_cfg["min_bound"]
        upper_bound = test_fun_cfg["max_bound"]
        d = self.config.solution_dimension
        self.minV = np.full(d, lower_bound)
        self.maxV = np.full(d, upper_bound)

    # 具体算法，返回最优值
    def _algorithm(self):
        pass

    # 运行标准优化测试函数
    def run_std_optim(self):
        self._set_bound()
        # 运行算法
        self._algorithm()
        # 保存迭代中的fitness
        # self.save_iteration_fitness_for_test_optim()
        # 保存FEs
        self.save_iteration_fes_for_test_optim()
        return self.best_value, self.best_solution