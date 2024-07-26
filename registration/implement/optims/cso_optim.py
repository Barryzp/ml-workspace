import numpy as np
from utils.tools import Tools
from optims.optim_base import Particle
from optims.ppso_optim import PPSO_optim

# 顶层loser只向winner学习
class Particle_CSO(Particle):
    def update_velocity(self, winner_pos):
        dim = self.pbest_position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        updated_loser_velocities = (random_coeff1 * self.velocity +
                                        random_coeff2 * (winner_pos - self.position) +
                                        random_coeff3 * (self.pbest_position - self.position))
        

        updated_loser_positions = self.position + updated_loser_velocities
        self.velocity = updated_loser_velocities
        self.position = updated_loser_positions

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

        fes = 0
        # 逻辑：排序，分层，选择，更新
        for _ in range(num_iterations * 2):
            check = self.check_match_finished()
            if check : return self.best_solution

            self.current_iterations = _

            # 主要是和其它算法比较这个少了一次的更新
            if _ % 2 == 1 :
                self.recording_data_item(_//2)
                self.recording_data_item_FEs(fes//2)
            
            fes += len(particles)
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
            
            # 更新粒子
            for index in range(separator):
                winner = winners[index]
                loser = losers[index]
                loser.update_velocity(winner.position)
                loser.evaluate()
                # 比较最大值
                self.set_best(winner, loser)

        # self.save_psos_parameters(particles, "end")
        return self.best_solution
