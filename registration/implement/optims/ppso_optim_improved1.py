import torch, math, random
import numpy as np
from utils.tools import Tools
from optims.ppso_optim import PPSO_optim, Particle

# ppso改进方案1：loser向任意上面的任意一层好粒子学，winner正常学
class Particle_PPSO1(Particle):
    # winner更新速度, 只需要upper_best
    def update_velocity_winner(self, upper_best, global_best, is_top = False):
        random_coeff1 = np.random.rand()
        random_coeff2 = np.random.rand()
        random_coeff3 = np.random.rand()
        
        updated_velocity = None
        updated_position = None
        # 非顶层更新速度
        if not is_top :
            random_coeff4 = np.random.rand()   
            updated_velocity = (random_coeff1 * self.velocity +
                        random_coeff2 * (upper_best - self.position) +
                        random_coeff3 * (self.best_position - self.position) +
                        self.pso_optim.phi * random_coeff4 * (global_best - self.position))
            updated_position = self.position + updated_velocity
        else:
            # 顶层不更新
            updated_velocity = self.velocity
            updated_position = self.position
        self.position = updated_position
        self.velocity = updated_velocity
        self.check()

    # loser更新速度，loser不仅朝着winner学，还朝着上层的winner学
    def update_velocity_loser(self, winner_pos, upper_pos):
        random_coeff1 = np.random.rand()
        random_coeff2 = np.random.rand()
        random_coeff3 = np.random.rand()
        random_coeff4 = np.random.rand()
        updated_loser_velocities = (random_coeff1 * self.velocity +
                                        random_coeff2 * (winner_pos - self.position) +
                                        random_coeff3 * (self.best_position - self.position) +
                                        random_coeff4 * (upper_pos - self.position))
        updated_loser_positions = self.position + updated_loser_velocities
        self.velocity = updated_loser_velocities
        self.position = updated_loser_positions
        self.check()


# 定义PPSO类，之后再将其改进
class PPSO_optim1(PPSO_optim):
    # 核心算法逻辑
    # PSO algorithm
    def _algorithm(self, particle_vals, num_iterations):
        particles = np.array([Particle_PPSO1(particle_vals[i], self, i) for i in range(len(particle_vals))])
        layers_num = len(self.layer_cfg)

        # 逻辑：排序，分层，选择，更新
        for _ in range(num_iterations):
            check = self.check_match_finished()
            if check : return self.best_solution

            self.current_iterations = _

            # 排序
            particles = self.sorted_particles(particles, True)
            self.recording_data_item(_, particles[0])

            # 在这之后就有best了
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
                top_indeces = np.random.permutation(separator) % top_layer_size
                aim_top_particles = top_layer_particles[top_indeces]

                is_top_layer = layer_idx == 0

                # 由于非顶层的winner会和上层的粒子进行混合，这里还需要得到上一层的粒子
                if not is_top_layer:
                    upper_particles = np.sum(self.layer_cfg[:layer_idx])
                    upper_indeces = np.random.choice(np.random.permutation(len(upper_particles)), size=separator)
                    aim_upper_particles = upper_particles[upper_indeces]

                # 更新粒子
                for index in range(separator):
                    winner = winners[index]
                    loser = losers[index]
                    if is_top_layer : 
                        winner.update_velocity_winner(None, None, is_top_layer)
                        loser.update_velocity_loser(winner.position, global_best.position)
                    else:
                        upper_best = aim_upper_particles[index]
                        global_best = aim_top_particles[index]
                        loser.update_velocity_loser(upper_best.position, global_best.position)
                        winner.update_velocity_winner(upper_best.position, global_best.position, is_top_layer)
                    
        # self.save_psos_parameters(particles, "end")
        return self.best_solution