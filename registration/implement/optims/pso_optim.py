import numpy as np
from utils.tools import Tools
from optims.optim_base import OptimBase, Particle

class PSO_optim(OptimBase):

    def __init__(self, config, share_records_out = None, matched_3dct_id = None) -> None:
        super(PSO_optim, self).__init__(config, share_records_out, matched_3dct_id)

    # 基本参数的初始化
    def init_basic_params(self):
        super(PSO_optim, self).init_basic_params()

        self.weight_inertia = self.config.weight_inertia  # Inertia weight
        self.individual_w = self.config.individual_w    # Cognitive (particle's best) weight
        self.global_w = self.config.global_w    # Social (swarm's best) weight
        self.speed_param_ratio = self.config.speed_param_ratio # 0.1 ~ 0.2
    
    def init_with_2d_params(self):
        super(PSO_optim, self).init_with_2d_params()
        # 每个粒子的移动速度是不同的, [speed_x, speed_y, speed_z, rotation_x, rotation_y, rotation_z]
        # 位移限制的范围
        translate_delta = self.config.translate_delta
        rotation_delta = self.config.rotation_delta[-1]
        speed_x = translate_delta[0] * self.speed_param_ratio
        speed_y = translate_delta[1] * self.speed_param_ratio
        speed_rotation = rotation_delta * self.speed_param_ratio
        self.speed = np.array([speed_x, speed_y, speed_rotation]) # 粒子移动的速度为参数范围的10%~20%

    def init_with_3d_params(self):
        super(PSO_optim, self).init_with_3d_params()

        # 位移限制的范围
        translate_delta = self.config.translate_delta
        rotation_delta = self.config.rotation_delta

        # 每个粒子的移动速度是不同的, [speed_x, speed_y, speed_z, rotation_x, rotation_y, rotation_z]
        speed_x = translate_delta[0] * self.speed_param_ratio
        speed_y = translate_delta[1] * self.speed_param_ratio
        speed_z = translate_delta[2] * self.speed_param_ratio

        speed_rot_x = rotation_delta[0] * 2 * self.speed_param_ratio
        speed_rot_y = rotation_delta[1] * 2 * self.speed_param_ratio
        speed_rot_z = rotation_delta[2] * self.speed_param_ratio

        self.speed = np.array([speed_x, speed_y, speed_z, speed_rot_x, speed_rot_y, speed_rot_z]) # 粒子移动的速度为参数范围的10%~20%

    # 需要进行调参，惯性权重，个体最优系数，种群最有系数
    # 采用环形边界的方式将点限制在一个范围内，
    # 但是这种环形边界是否合理呢？还是采用弹性边界？
    # 其实问题也不是很大，因为都要向着最优的方向移动，弹性边界的优势在于
    # 最优区域在边界的时候，那么收敛较快些，但容易陷入局部最优，在边界震荡。
    # 但咱们的这个方法不是特别需要，更需要的是在整个参数空间中均匀搜索
    
    # 循环边界处理
    def loop_boundary_constrain(self, t):
        range_size = self.maxV - self.minV
        position = self.minV + np.remainder(t - self.minV, range_size)
        return position

    # 反射边界处理，改变粒子的运动方向，镜像反射
    def reflect_boundary_constrain(self, t):
        for i in range(len(t)):
            if t[i] < self.minV[i]:
                t[i] = 2 * self.minV[i] - t[i]
                if t[i] > self.maxV[i]:  # 防止超出上边界
                    t[i] = self.maxV[i]
            elif t[i] > self.maxV[i]:
                t[i] = 2 * self.maxV[i] - t[i]
                if t[i] < self.minV[i]:  # 防止超出下边界
                    t[i] = self.minV[i]
        return t

    # 反弹边界处理
    def bounce_boundary(self, t):
        for i in range(len(t)):
            if t[i] < self.minV[i]:
                t[i] = self.minV[i] + (self.minV[i] - t[i])
                if t[i] > self.maxV[i]:  # 防止超出上边界
                    t[i] = self.maxV[i]
            elif t[i] > self.maxV[i]:
                t[i] = self.maxV[i] - (t[i] - self.maxV[i])
                if t[i] < self.minV[i]:  # 防止超出下边界
                    t[i] = self.minV[i]
        return t

    # 随机重置边界
    # def random_reset_constrain(self, t):
    #     for i in range(len(t)):
    #         if t[i] < self.minV[i] or t[i] > self.maxV[i]:
    #             t[i] = self.minV[i] + XX.rand(1).item() * (self.maxV[i] - self.minV[i])
    #     return t

    def force_clip_position(self, t):
        return np.clip(t, self.minV, self.maxV)

    def constrain(self, t):
        return self.force_clip_position(t)
        # item_num = t.numel()
        # def judge(i):
        #     return t[i] < self.minV[i] or t[i] > self.maxV[i]
        
        # def fixed(i):
        #     if t[i] < self.minV[i] or t[i] > self.maxV[i]:
        #         t[i] = self.minV[i] + t[i] % (self.maxV[i] - self.minV[i])

        # # 只限定x,y,也要限定z旋转角度z
        # for index in range(item_num):
        #     while(judge(index)):
        #         fixed(index)

        # return t

    # 将速度限制在最大范围内
    def constrain_velocity(self, velocity):
        if self.config.mode == "test" : return velocity
        max_velocity = self.speed
        return np.clip(velocity, -max_velocity, max_velocity)

    def _get_init_gbest(self):
        particle_num = self.config.particle_num
        particles = [Particle(self, i) for i in range(particle_num)]
        gbest_particle = max(particles, key=lambda p: p.pbest_value)
        gbest_value = gbest_particle.pbest_value
        gbest_position = gbest_particle.pbest_position
        self.set_best(gbest_value, gbest_position)
        return gbest_position, gbest_value, particles

    # PSO algorithm
    def _algorithm(self):
        gbest_position, gbest_value, particles = self._get_init_gbest()

        while not self.check_end():
            check = self.check_match_finished()
            if check : return gbest_position
            for particle in particles:
                self.recording_data_item_FEs()
                particle.update(gbest_position)
                particle.evaluate()
                if particle.pbest_value > gbest_value:
                    gbest_value = particle.pbest_value
                    gbest_position = particle.pbest_position
                    self.set_best(gbest_value, gbest_position)
                self.add_fes()
                if self.check_end():
                    break

        # self.save_psos_parameters(particles, "end")
        return gbest_position

    # 保存pso的所有参数
    def save_psos_parameters(self, psos, prefix = ""):
        if self.config.mode != "matched": return

        records = []
        file_path = Tools.get_save_path(self.config)
        file_name = f"{self.ct_matching_slice_index}_{prefix}_particle_pos.csv"
        columns = ["x", "y", "rotation", "v1", "v2", "v3"]
        for particle in psos:
            position = particle.position
            velocity = particle.velocity
            data_item = np.concatenate((position, velocity), axis=0)
            records.append(data_item.tolist())
            
        Tools.save_params2df(records, columns, file_path, file_name)

        # 进行优化
    def run(self):
        super(PSO_optim, self).run()

        self.spawn_random_particles()
        # Running PSO
        iter_best_position = self._algorithm()
        fit_res = self.fitness(iter_best_position)

        val, best_regi_img, best_slice = fit_res[0], fit_res[1],fit_res[2]

        if self.config.mode == "matched":
            self.ct_matching_slice_index = best_slice
            self.put_best_data_in_share(self.matched_3dct_id, iter_best_position, val, best_slice)
            print(f"The current slice index found is: {self.ct_matching_slice_index}")
            print(f"The current iteration best position found is: {iter_best_position}")
            print(f"The cur iteration maximum value of the function is: {val}")
            # 保存相关数据(图像之类的)
            self.save_iteration_best_reg_img(best_regi_img, self.config.iteratons)
        else:
            print(f"The current iteration best position found is: {iter_best_position}")
            print(f"The cur iteration maximum value of the function is: {val}")
            self.save_iteration_params()

        return val, best_regi_img, iter_best_position