import numpy as np
from utils.tools import Tools
from optims.optim_base import Particle
from optims.ppso_optim import PPSO_optim

# 顶层loser只向winner学习
class Particle_CSO(Particle):
    def update_velocity(self, winner_pos, swarm_mean_pos):
        # 还有一项，那就是平均位置
        dim = self.pbest_position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        random_coeff4 = np.random.rand(dim)
        # 原始CSO并不需要个体最优
        updated_loser_velocities = (random_coeff1 * self.velocity +
                                        random_coeff2 * (winner_pos - self.position) +
                                        # random_coeff3 * (self.pbest_position - self.position) +
                                        random_coeff4 * (swarm_mean_pos - self.position))

        updated_loser_positions = self.position + updated_loser_velocities
        self.velocity = updated_loser_velocities
        self.position = updated_loser_positions
        self.check()

# loser学习， winner不学习
class CSO_optim(PPSO_optim):
    # 核心算法逻辑
    # PSO algorithm
    def _algorithm(self):
        particle_num = self.config.particle_num
        num_iterations = self.config.iteratons
        particles = np.array([Particle_CSO(self, i) for i in range(particle_num)])
        gbest_particle = max(particles, key=lambda p: p.pbest_value)
        self.set_best(gbest_particle, gbest_particle)
        self.recording_data_item_FEs()

        # 逻辑：排序，分层，选择，更新
        while not self.check_end():
            check = self.check_match_finished()
            if check : return self.best_solution

            # 配对
            rand_indeces = np.random.permutation(particle_num)
            # 分成俩部分
            separator = particle_num // 2
            # 构成配对
            rand_pairs = np.column_stack((rand_indeces[:separator], rand_indeces[separator:2 * separator]))
            comparison_mask = [
                    particles[rand_pairs[i, 0]].current_fitness > particles[rand_pairs[i, 1]].current_fitness
                    for i in range(len(rand_pairs))]
            winner_indeces = np.where(comparison_mask, rand_pairs[:, 0], rand_pairs[:, 1])
            loser_indeces = np.where(comparison_mask, rand_pairs[:, 1], rand_pairs[:, 0])
            losers = particles[loser_indeces]
            winners = particles[winner_indeces]
            
            # 得到种群的平均位置
            mean_pos = np.zeros_like((self.best_solution))
            for particle in particles:
                mean_pos += particle.position 
            mean_pos = mean_pos / particle_num            

            # 更新粒子
            for index in range(separator):
                winner = winners[index]
                loser = losers[index]
                loser.update_velocity(winner.position, mean_pos)
                loser.evaluate()
                self.add_fes()
                # 比较最大值
                self.set_best(winner, loser)
                self.recording_data_item_FEs()

        # self.save_psos_parameters(particles, "end")
        return self.best_solution
