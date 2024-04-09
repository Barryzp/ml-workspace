import torch, math, random
import numpy as np
from utils.tools import Tools

# Particle class
class Particle:
    def __init__(self, x0, pso_optim):
        self.position = x0
        self.pso_optim = pso_optim

        self.velocity = torch.rand_like(x0)
        self.best_position = x0
        self.best_value,_ = pso_optim.fitness(x0)

    def update_velocity(self, global_best_position):
        r1 = random.random()
        r2 = random.random()
        individual_w = self.pso_optim.individual_w
        global_w = self.pso_optim.global_w
        weight_inertia = self.pso_optim.weight_inertia

        cog_dir = torch.zeros_like(global_best_position)
        cog_delta = (self.best_position - self.position)
        cog_norm = torch.norm(cog_delta)
        if cog_norm > 0:
            cog_dir = cog_delta / cog_norm 
        cognitive_velocity = individual_w * r1 * cog_dir
        
        soc_dir = torch.zeros_like(global_best_position)
        soc_delta = (global_best_position - self.position)
        soc_norm = torch.norm(soc_delta)
        if soc_norm > 0:
            soc_dir = soc_delta / soc_norm
        social_velocity = global_w * r2 * soc_dir
        self.velocity = weight_inertia * self.velocity + cognitive_velocity + social_velocity

    def move(self):
        speed = self.pso_optim.speed

        self.position += self.velocity * speed
        # constrain the position in a range
        self.position = self.pso_optim.constrain(self.position)

        value,_ = self.pso_optim.fitness(self.position)

        if value > self.best_value:
            self.best_position = self.position
            self.best_value = value

class PSO_optim:

    def __init__(self, config) -> None:
        self.config = config

    def set_init_params(self, refer_img_size, reg_similarity):
        self.init_basic_params()
        self.reg_similarity = reg_similarity
        if self.config.mode == "2d":
            self.init_with_2d_params()
        if self.config.mode == "3d":
            width, height = refer_img_size
            if height > width:
                border = width
            else:
                border = height
            self.init_with_3d_params(border)

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

        init_translate = (814.0, 804.0)
        init_rotation = 0.0
        # 位移限制的范围
        translate_delta = 20.0, 15.0
        rotation_delta = 10

        # 生成初始参数规定范围，
        self.minV = [
                init_translate[0] - translate_delta[0], 
                init_translate[1] - translate_delta[1],
                init_rotation - rotation_delta, 
        ]
        self.maxV = [
            init_translate[0] + translate_delta[0], 
                init_translate[1] + translate_delta[1],
                init_rotation + rotation_delta, 
        ]

        # 每个粒子的移动速度是不同的, [speed_x, speed_y, speed_z, rotation_x, rotation_y, rotation_z]
        speed_x = translate_delta[0] * 2 * self.speed_param_ratio
        speed_y = translate_delta[1] * 2 * self.speed_param_ratio
        speed_rotation = rotation_delta * 2 * self.speed_param_ratio
        self.speed = torch.tensor([speed_x, speed_y, speed_rotation]) # 粒子移动的速度为参数范围的10%~20%


    # 2d/3d图像的最优参数查找
    def init_with_3d_params(self, img_border_len):
        ############## PSO参数，六个（translate_x,y,z, rotation_x,y,z）###################
        self.slice_num = 40
        self.parameters_num = 6
        
        # HACK 这个坐标需要注意，是需要记录的
        init_translate = (814.0, 804.0, self.slice_num * 0.5)
        init_rotation = (0.0, 0.0, 0.0)
        # 位移限制的范围
        translate_delta = (20.0, 15.0, 10.0)
        rotation_delta = math.degrees(math.atan((self.slice_num * 0.5)/(img_border_len * 0.5))) 
        self.rotation_center_xy = (960.0, 960.0)
        # 生成初始参数规定范围，
        self.minV = [
            init_translate[0] - translate_delta[0], 
                init_translate[1] - translate_delta[1],
                init_translate[2] - translate_delta[2],
                init_rotation[0] - rotation_delta, 
                init_rotation[1] - rotation_delta,
                init_rotation[2] - rotation_delta,
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
        item_num = t.numel()
        if t[0] < 0 or t[1] < 0:
            print(f"x less zero! item_num: {item_num}")

        for i in range(item_num):
            if t[i] < self.minV[i]:
                t[i] = self.maxV[i] - (self.minV[i] - t[i])
            elif t[i] > self.maxV[i]:
                t[i] = self.minV[i] + (t[i] - self.maxV[i])
        return t

    # PSO algorithm
    def _algorithm(self, particle_vals, num_iterations, record):
        particles = [Particle(particle_vals[i], self) for i in range(len(particle_vals))]
        global_best_position = max(particles, key=lambda p: p.best_value).position.clone()

        for _ in range(num_iterations):
            global_best_val, __ = self.fitness(global_best_position)
            if record != None:
                data_item = global_best_position.numpy()
                data_item = np.insert(data_item, 0, _)
                data_item = np.insert(data_item, data_item.size, global_best_val)
                record.append(data_item.tolist())
                self.save_iteration_best_reg_img(__, _)

            print(f"iterations: {_}, fitness: {global_best_val}, params: {global_best_position}")
            local_best = global_best_val
            for particle in particles:
                particle.update_velocity(global_best_position)
                particle.move()
                if particle.best_value > local_best:
                    local_best = particle.best_value
                    global_best_position = particle.best_position.clone()

        return global_best_position

    def fitness(self, position):
        return self.reg_similarity(position)

    # 保存迭代过程中的参数
    def save_iteration_params(self, records):
        if self.config.mode == "2d":
            columns = ["iterations", "x", "y", "rotation", "fitness"]
        elif self.config.mode == "3d":
            columns = ["iterations", 
                       "x", "y", "z", 
                       "rotation_x", "rotation_y", "rotation_z",
                       "fitness"]
        file_path = f"{self.config.data_save_path}/{self.config.record_id}"
        Tools.save_params2df(records, columns, file_path, f"pso_params_{self.config.mode}.csv")

    def save_iteration_best_reg_img(self, img_array, iterations):
        img_path = f"{self.config.data_save_path}/{self.config.record_id}"
        file_name = f"iter{iterations}_best_reg.bmp"
        Tools.save_img(img_path, file_name, img_array)

        # 进行优化
    def run(self):
        poses = [torch.tensor([random.random() * (self.maxV[j] - self.minV[j]) + self.minV[j] for j in range(self.parameters_num)]) for i in range(self.particle_num)]
        records = []
        # Running PSO
        best_position = self._algorithm(poses, self.iteratons, records)
        self.save_iteration_params(records)
        print(f"The best position found is: {best_position}")
        val, best_regi_img = self.fitness(best_position)
        print(f"The maximum value of the function is: {val}")
        return val, best_regi_img