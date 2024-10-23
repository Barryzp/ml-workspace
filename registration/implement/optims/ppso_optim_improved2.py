import numpy as np
from utils.tools import Tools
from optims.ppso_optim import PPSO_optim, Particle_PPSO

# ppso改进方案4：loser向任意上面的任意一层好粒子学，winner正常学，
# 大于一定迭代次数之后loser还朝着上方学习
class Particle_PPSO4(Particle_PPSO):

    # winner更新速度, 想着上层学
    def update_velocity_winner(self, upper_pos, is_top = False):
        dim = self.position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        
        updated_velocity = None
        updated_position = None
        # 非顶层更新速度
        if not is_top :
            updated_velocity = random_coeff1 * self.velocity + \
            random_coeff2 * (upper_pos - self.position) + \
            random_coeff3 * (self.pbest_position - self.position)
            updated_position = self.position + updated_velocity
        else:
            # 顶层不更新
            updated_velocity = self.velocity
            updated_position = self.position
        self.position = updated_position
        self.velocity = updated_velocity
        self.check()

    # loser更新速度，loser朝着上任意一层学习，顶层loser朝着winner学
    def update_velocity_loser(self, winner_pos, upper_pos, swtich_strategy = False, is_top_layer=False):
        dim = self.position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        random_coeff4 = np.random.rand(dim)
        # 前期朝着winner学，后期开始收缩
        updated_loser_velocities = (random_coeff1 * self.velocity 
                                    + random_coeff3 * (self.pbest_position - self.position)
                                    + random_coeff4 * (winner_pos - self.position))
        if swtich_strategy and not is_top_layer:
            random_coeff2 = np.random.rand(dim)
            updated_loser_velocities += random_coeff2 * (upper_pos - self.position)
        updated_loser_positions = self.position + updated_loser_velocities
        self.velocity = updated_loser_velocities
        self.position = updated_loser_positions
        self.check()

class PPSO_optim4(PPSO_optim):

    # 判断是否达到切换搜索策略
    def judge_switch_strategy(self):
        current_fes = self.get_fes()
        max_fes = self.config.max_fes
        ratio = current_fes / max_fes
        return ratio > self.config.switch_strategy_ratio

    # 核心算法逻辑
    # PSO algorithm
    def _algorithm(self):
        particle_num = self.config.particle_num
        particles = np.array([Particle_PPSO4(self, i) for i in range(particle_num)])
        gbest_particle = max(particles, key=lambda p: p.pbest_value)
        self.set_best(gbest_particle, gbest_particle)
        self.recording_data_item_FEs()

        layers_num = len(self.layer_cfg)
        cumulative_layers = np.cumsum(self.layer_cfg)

        # 逻辑：排序，分层，选择，更新
        while not self.check_end():
            check = self.check_match_finished()
            if check : return self.best_solution

            # 排序
            particles = self.sorted_particles(particles, True)
            # 在这之后就有best了
            # 分层：适应值的倒序数组就是分层结构，咱们看成就行了
            # 构造金字塔，咱们这个结构不是并行化就没有必要进行原始代码中的构造数组
            # 配对：金字塔从底层开始往上面回溯

            # 构造一个预定义的金字塔，方便取
            particle_pyramid = [[] for _ in range(layers_num)]
            for i in range(layers_num):
                if i == 0:
                    indices = range(cumulative_layers[0])
                else:
                    indices = range(cumulative_layers[i - 1], cumulative_layers[i])
                for j in indices:
                    particle_pyramid[i].append(particles[j])
                particle_pyramid[i] = np.array(particle_pyramid[i])

            for layer_idx in range(layers_num - 1, -1, -1):
                layer_size = self.layer_cfg[layer_idx]
                # 索引对应层数的粒子
                layer_particles = particle_pyramid[layer_idx]

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

                is_top_layer = layer_idx == 0

                should_switch = self.judge_switch_strategy()

                # 由于非顶层的winner会和上层的粒子进行混合，这里还需要得到上一层的粒子
                if not is_top_layer:
                    if not should_switch:
                        # 随机选取上层的粒子
                        upper_particles = particle_pyramid[layer_idx-1]
                    else:
                        # 随机选取某一层的最佳，往这个方向走
                        random_layer_idx = np.random.choice(range(layer_idx), size=1)[0]
                        upper_particles = particle_pyramid[random_layer_idx]
                    upper_particle_size = len(upper_particles)
                    upper_indeces_winner = self.random_choice_indeces(upper_particle_size, separator)
                    upper_indeces_loser = self.random_choice_indeces(upper_particle_size, separator)
                    upper_particles_winner = upper_particles[upper_indeces_winner]
                    upper_particles_loser = upper_particles[upper_indeces_loser]

                # 更新粒子
                for index in range(separator):
                    winner = winners[index]
                    loser = losers[index]
                    if is_top_layer : 
                        winner.update_velocity_winner(None, is_top_layer)
                        loser.update_velocity_loser(winner.position, None, should_switch, is_top_layer)
                    else:
                        upper_best_winner = upper_particles_winner[index]
                        upper_best_loser = upper_particles_loser[index]
                        loser.update_velocity_loser(winner.position, upper_best_loser.position, should_switch, is_top_layer)
                        winner.update_velocity_winner(upper_best_winner.position, is_top_layer)
                        winner.evaluate()
                        self.add_fes()
                    loser.evaluate()
                    self.add_fes()
                    self.set_best(winner, loser)
                    self.recording_data_item_FEs()

        # self.save_psos_parameters(particles, "end")
        return self.best_solution


# ppso改进方案5：之前是正常的PPSO，
# 大于一定迭代次数后，loser向任意上面的任意一层好粒子学，再加上一个随机扩散速度来增加粒子的多样性
class Particle_PPSO5(Particle_PPSO4):
    # winner更新速度, 想着上层学
    def update_velocity_winner(self, upper_pos, swtich_strategy = False, is_top = False):
        dim = self.position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        random_coeff4 = np.random.rand(dim)

        updated_velocity = None
        updated_position = None


        # 非顶层更新速度
        if not is_top :
            updated_velocity = (random_coeff1 * self.velocity + 
            random_coeff2 * (upper_pos - self.position) + 
            random_coeff3 * (self.pbest_position - self.position))
            updated_position = self.position + updated_velocity
        else:
            # 顶层不更新
            updated_velocity = self.velocity
            updated_position = self.position
        self.position = updated_position
        self.velocity = updated_velocity
        self.check()

    def update_velocity_loser(self, winner_pos, upper_pos, swtich_strategy = False, is_top_layer=False):
        dim = self.position.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)
        random_coeff4 = np.random.rand(dim)

        random_part = np.zeros_like(self.position)
        if not swtich_strategy : random_part = self.optim.gen_random_position()

        if swtich_strategy and not is_top_layer:
            updated_loser_velocities = (random_coeff1 * self.velocity 
                                    + random_coeff3 * (self.pbest_position - self.position)
                                    + random_coeff4 * (upper_pos - self.position))
        else:
            remain_part = self.optim.config.random_part_weight * random_coeff2 * (random_part - self.position)
            updated_loser_velocities = (random_coeff1 * self.velocity 
                                    + random_coeff3 * (self.pbest_position - self.position)
                                    + random_coeff4 * (winner_pos - self.position)
                                    + remain_part)
        updated_loser_positions = self.position + updated_loser_velocities
        self.velocity = updated_loser_velocities
        self.position = updated_loser_positions
        self.check()


class PPSO_optim5(PPSO_optim4):

    # 核心算法逻辑
    # PSO algorithm
    def _algorithm(self):
        particle_num = self.config.particle_num
        particles = np.array([Particle_PPSO5(self, i) for i in range(particle_num)])
        gbest_particle = max(particles, key=lambda p: p.pbest_value)
        self.set_best(gbest_particle, gbest_particle)
        self.recording_data_item_FEs()

        layers_num = len(self.layer_cfg)
        cumulative_layers = np.cumsum(self.layer_cfg)

        # 逻辑：排序，分层，选择，更新
        while not self.check_end():
            check = self.check_match_finished()
            if check : return self.best_solution

            # 排序
            particles = self.sorted_particles(particles, True)
            # 在这之后就有best了
            # 分层：适应值的倒序数组就是分层结构，咱们看成就行了
            # 构造金字塔，咱们这个结构不是并行化就没有必要进行原始代码中的构造数组
            # 配对：金字塔从底层开始往上面回溯

            # 构造一个预定义的金字塔，方便取
            particle_pyramid = [[] for _ in range(layers_num)]
            for i in range(layers_num):
                if i == 0:
                    indices = range(cumulative_layers[0])
                else:
                    indices = range(cumulative_layers[i - 1], cumulative_layers[i])
                for j in indices:
                    particle_pyramid[i].append(particles[j])
                particle_pyramid[i] = np.array(particle_pyramid[i])

            for layer_idx in range(layers_num - 1, -1, -1):
                layer_size = self.layer_cfg[layer_idx]
                # 索引对应层数的粒子
                layer_particles = particle_pyramid[layer_idx]

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

                is_top_layer = layer_idx == 0

                should_switch = self.judge_switch_strategy()

                # 由于非顶层的winner会和上层的粒子进行混合，这里还需要得到上一层的粒子
                if not is_top_layer:
                    if not should_switch:
                        # 随机选取上层的粒子
                        upper_particles = particle_pyramid[layer_idx-1]
                    else:
                        # 随机选取某一层的最佳，往这个方向走
                        random_layer_idx = np.random.choice(range(layer_idx), size=1)[0]
                        upper_particles = particle_pyramid[random_layer_idx]
                    upper_particle_size = len(upper_particles)
                    upper_indeces_winner = self.random_choice_indeces(upper_particle_size, separator)
                    upper_indeces_loser = self.random_choice_indeces(upper_particle_size, separator)
                    upper_particles_winner = upper_particles[upper_indeces_winner]
                    upper_particles_loser = upper_particles[upper_indeces_loser]

                # 更新粒子
                for index in range(separator):
                    winner = winners[index]
                    loser = losers[index]
                    if is_top_layer : 
                        winner.update_velocity_winner(None, should_switch, is_top_layer)
                        loser.update_velocity_loser(winner.position, None, should_switch, is_top_layer)
                    else:
                        upper_best_winner = upper_particles_winner[index]
                        upper_best_loser = upper_particles_loser[index]
                        loser.update_velocity_loser(winner.position, upper_best_loser.position, should_switch, is_top_layer)
                        winner.update_velocity_winner(upper_best_winner.position, should_switch, is_top_layer)
                        winner.evaluate()
                        self.add_fes()
                    loser.evaluate()
                    self.add_fes()
                    self.set_best(winner, loser)
                    self.recording_data_item_FEs()

        # self.save_psos_parameters(particles, "end")
        return self.best_solution
