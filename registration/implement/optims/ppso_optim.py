import numpy as np
from utils.tools import Tools
from optims.optim_base import Particle
from optims.pso_optim import PSO_optim

# Particle class
class Particle_PPSO(Particle):

    # winner更新速度
    def update_velocity_winner(self, upper_best, global_best, is_top = False):
        dim = self.position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        
        updated_velocity = None
        updated_position = None
        # 非顶层更新速度
        if not is_top :
            random_coeff4 = np.random.rand(dim)   
            updated_velocity = (random_coeff1 * self.velocity +
                        random_coeff2 * (upper_best - self.position) +
                        random_coeff3 * (self.pbest_position - self.position) +
                        self.optim.phi * random_coeff4 * (global_best - self.position))
            updated_position = self.position + updated_velocity
        else:
            # 顶层不更新
            updated_velocity = self.velocity
            updated_position = self.position
        self.position = updated_position
        self.velocity = updated_velocity
        self.check()
    
    # loser更新速度
    def update_velocity_loser(self, winner_pos):
        dim = self.position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        updated_loser_velocities = (random_coeff1 * self.velocity +
                                        random_coeff2 * (winner_pos - self.position) +
                                        random_coeff3 * (self.pbest_position - self.position))
        updated_loser_positions = self.position + updated_loser_velocities
        self.velocity = updated_loser_velocities
        self.position = updated_loser_positions
        self.check()

# 定义PPSO类，之后再将其改进
class PPSO_optim(PSO_optim):
    # 没有惯性权重，个体权重，群体权重系数
    def __init__(self, config, share_records_out = None, matched_3dct_id = None) -> None:
        super(PPSO_optim, self).__init__(config, share_records_out, matched_3dct_id)

        # 初始化layer配置，数组形式:[4, 8, 20, 32]
        self.layer_cfg = config.layer_config
        self.phi = config.phi
        
    # 基本参数的初始化
    def init_basic_params(self):
        super(PPSO_optim, self).init_basic_params()
        # 粒子是这样分的
        self.particle_num = np.sum(self.layer_cfg)

    def sorted_particles(self, particles, reverse=False):
        coeff = 1
        if reverse : coeff = -1
        # 或者，直接使用负数索引和步长-1来反转
        sorted_indices_desc = np.argsort(coeff * np.array([particle.current_fitness for particle in particles]))
        # 使用排序索引重新排序数组
        sorted_objects_desc = particles[sorted_indices_desc]
        return sorted_objects_desc

    # 核心算法逻辑
    # PSO algorithm
    def _algorithm(self):
        particle_num = self.config.particle_num
        particles = np.array([Particle_PPSO(self, i) for i in range(particle_num)])
        gbest_particle = max(particles, key=lambda p: p.pbest_value)
        super().set_best(gbest_particle.pbest_value, gbest_particle.pbest_position)
        self.recording_data_item_FEs()

        layers_num = len(self.layer_cfg)

        # 逻辑：排序，分层，选择，更新
        while not self.check_end():
            check = self.check_match_finished()
            if check : return self.best_solution

            # 这里需要注意了，PPSO使用的粒子的当前适应值来进行排序的，那么历史最优就不是当前的这个粒子的最优了
            # 排序
            particles = self.sorted_particles(particles, True)
            # 分层：适应值的倒序数组就是分层结构，咱们看成就行了
            # 构造金字塔，咱们这个结构不是并行化就没有必要进行原始代码中的构造数组
            # 配对：金字塔从底层开始往上面回溯
            current_moved_particles = 0
            for layer_idx in range(layers_num - 1, -1, -1):
                layer_size = self.layer_cfg[layer_idx]

                # 索引对应层数的粒子
                if current_moved_particles == 0:
                    layer_particles = particles[-layer_size:]
                else:
                    start_idx = layer_size + current_moved_particles
                    layer_particles = particles[-start_idx:-current_moved_particles]
                current_moved_particles += layer_size

                rand_indeces = np.random.permutation(layer_size)
                # 分成俩部分
                separator = layer_size // 2
                # 构成配对
                rand_pairs = np.column_stack((rand_indeces[:separator], rand_indeces[separator:2 * separator]))
                comparison_mask = [
                    layer_particles[rand_pairs[i, 0]].current_fitness > layer_particles[rand_pairs[i, 1]].current_fitness
                    for i in range(len(rand_pairs))]
                
                winner_indeces = np.where(comparison_mask, rand_pairs[:, 0], rand_pairs[:, 1])
                loser_indeces = np.where(comparison_mask, rand_pairs[:, 1], rand_pairs[:, 0])
                losers = layer_particles[loser_indeces]
                winners = layer_particles[winner_indeces]

                # 获取最顶层粒子
                top_layer_size = self.layer_cfg[0]
                top_layer_particles = particles[0:top_layer_size]
                top_indeces = np.random.choice(np.random.permutation(top_layer_size), size=separator)
                aim_top_particles = top_layer_particles[top_indeces]

                is_top_layer = layer_idx == 0

                # 由于非顶层的winner会和上层的粒子进行混合，这里还需要得到上一层的粒子
                if not is_top_layer:
                    upper_layer_size = self.layer_cfg[layer_idx - 1]
                    start_idx = np.sum(self.layer_cfg[layer_idx-1:])
                    # 应该索引到当前的粒子数量
                    upper_layer_particles = particles[-start_idx:-layer_size]
                    upper_indeces = np.random.choice(np.random.permutation(upper_layer_size), size=separator)
                    aim_upper_particles = upper_layer_particles[upper_indeces]

                # 更新粒子
                for index in range(separator):
                    winner = winners[index]
                    loser = losers[index]
                    # 先更新loser
                    loser.update_velocity_loser(winner.position)
                    loser.evaluate()
                    self.add_fes()

                    # winner根据情况更新
                    if is_top_layer : winner.update_velocity_winner(None, None, is_top_layer)
                    else:
                        upper_best = aim_upper_particles[index]
                        global_best = aim_top_particles[index]
                        winner.update_velocity_winner(upper_best.position, global_best.position, is_top_layer)
                        winner.evaluate()
                        self.add_fes()

                    # 比较粒子的最大适应值，然后保存
                    self.set_best(winner, loser)
                    self.recording_data_item_FEs()

        # self.save_psos_parameters(particles, "end")
        return self.best_solution

    def set_best(self, winner, loser):
        if winner.pbest_value > loser.pbest_value:
            super().set_best(winner.pbest_value, winner.pbest_position)
        else:
            super().set_best(loser.pbest_value, loser.pbest_position)