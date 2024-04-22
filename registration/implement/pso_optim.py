import torch, math, random
import numpy as np
from scipy.stats import qmc
from utils.tools import Tools

# Particle class
class Particle:
    def __init__(self, x0, pso_optim, id):
        self.id = id
        self.position = x0
        self.pso_optim = pso_optim

        self.velocity = torch.rand_like(x0)
        self.best_position = torch.clone(x0)
        
        self.debug = False

        fit_res = pso_optim.fitness(x0)
        self.best_value = fit_res[0]

    def update_velocity(self, global_best_position):
        r1 = random.random()
        r2 = random.random()
        individual_w = self.pso_optim.individual_w
        global_w = self.pso_optim.global_w
        weight_inertia = self.pso_optim.weight_inertia

        cog_delta = (self.best_position - self.position)
        # cog_dir = torch.zeros_like(global_best_position)
        # cog_norm = torch.norm(cog_delta)
        # if cog_norm > 0:
        #     cog_dir = cog_delta / cog_norm
        cognitive_velocity = individual_w * r1 * cog_delta


        soc_delta = (global_best_position - self.position)
        # soc_dir = torch.zeros_like(global_best_position)
        # soc_norm = torch.norm(soc_delta)
        # if soc_norm > 0:
        #     soc_dir = soc_delta / soc_norm
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

class PSO_optim:

    def __init__(self, config, share_records_out = None) -> None:
        # 匹配过程中的ct切片索引
        self.ct_matching_slice_index = None
        self.share_records_out = share_records_out
        self.current_iterations = 0
        self.config = config

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

        self.weight_inertia = self.config.weight_inertia  # Inertia weight
        self.individual_w = self.config.individual_w    # Cognitive (particle's best) weight
        self.global_w = self.config.global_w    # Social (swarm's best) weight
        self.speed_param_ratio = self.config.speed_param_ratio # 0.1 ~ 0.2

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
        ############## PSO参数，六个（translate_x,y,z, rotation_x,y,z）###################
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

    # 需要进行调参，惯性权重，个体最优系数，种群最有系数
    # 采用环形边界的方式将点限制在一个范围内
    def constrain(self, t):
        # 这里只循环了一次，需要多次处理
        # item_num = t.numel()
        def judge(i):
            return t[i] < self.minV[i] or t[i] > self.maxV[i]
        
        def fixed(i):
            if t[i] < self.minV[i] or t[i] > self.maxV[i]:
                t[i] = self.minV[i] + t[i] % (self.maxV[i] - self.minV[i])

        # 只限定x,y
        for index in range(2):
            while(judge(index)):
                fixed(index)

        return t

    # 将速度限制在最大范围内
    def constrain_velocity(self, velocity):
        max_velocity = self.speed
        return torch.clip(velocity, -max_velocity, max_velocity)

    # PSO algorithm
    def _algorithm(self, particle_vals, num_iterations, record):
        particles = [Particle(particle_vals[i], self, i) for i in range(len(particle_vals))]
        global_best_position = max(particles, key=lambda p: p.best_value).position.clone()
        
        self.save_psos_parameters(particles, "start")

        matched_suffix = ""
        if self.config.mode == "matched":
            matched_suffix = f" slice_index: {self.ct_matching_slice_index},"

        for _ in range(num_iterations):
            self.current_iterations = _

            fit_res = self.fitness(global_best_position)
            global_best_val = fit_res[0]
            __ = fit_res[1]
            weighted_sp = fit_res[-1]
            
            if record != None:
                data_item = global_best_position.numpy()
                data_item = np.insert(data_item, 0, _)
                data_item = np.insert(data_item, data_item.size, global_best_val)
                record.append(data_item.tolist())
                if self.config.mode != "matched": self.save_iteration_best_reg_img(__, _)

            print(f"iterations: {_}, fitness: {global_best_val}, weighted_sp: {weighted_sp},{matched_suffix} params: {global_best_position}")
            local_best = global_best_val
            for particle in particles:
                particle.update_velocity(global_best_position)
                particle.move()
                if particle.best_value > local_best:
                    local_best = particle.best_value
                    global_best_position = particle.best_position.clone()

        self.save_psos_parameters(particles, "end")
        return global_best_position

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
        file_name = f"pso_params_{self.config.mode}.csv"

        if self.config.mode == "2d" :
            columns = ["iterations", "x", "y", "rotation", "fitness"]
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

    # 保存pso的所有参数
    def save_psos_parameters(self, psos, prefix = ""):
        if self.config.mode != "matched": return

        file_path = Tools.get_save_path(self.config)
        file_name = f"{self.ct_matching_slice_index}_{prefix}_particle_pos.csv"
        columns = ["x", "y", "rotation", "v1", "v2", "v3"]
        records = []
        for particle in psos:
            position = particle.position
            velocity = particle.velocity
            data_item = torch.concat((position, velocity), dim=0)
            records.append(data_item.tolist())
            
        Tools.save_params2df(records, columns, file_path, file_name)

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

        return torch.tensor(scaled_sample)
    
    def spawn_random_particles(self):
        return [torch.tensor([random.random() * (self.maxV[j] - self.minV[j]) + self.minV[j] for j in range(self.parameters_num)]) for i in range(self.particle_num)]

        # 进行优化
    def run(self):
        poses = self.spawn_uniform_particles()#[torch.tensor([random.random() * (self.maxV[j] - self.minV[j]) + self.minV[j] for j in range(self.parameters_num)]) for i in range(self.particle_num)]
        records = []

        # Running PSO
        best_position = self._algorithm(poses, self.iteratons, records)
        print(f"The best position found is: {best_position}")
        fit_res = self.fitness(best_position)
        val, best_regi_img = fit_res[0], fit_res[1]
        print(f"The maximum value of the function is: {val}")
        self.put_best_data_in_share(fit_res, best_position)
        self.save_iteration_params(records)
        if self.config.mode == "matched":
            # 保存相关数据(图像之类的)
            self.save_iteration_best_reg_img(best_regi_img, self.config.iteratons)

        return val, best_regi_img