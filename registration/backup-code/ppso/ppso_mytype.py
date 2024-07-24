import numpy as np

class Particle_PPSO:
    def __init__(self, position, velocity, best_position, best_value, pso_optim):
        self.position = position
        self.velocity = velocity
        self.best_position = position
        fit = pso_optim.fitness(position)
        self.best_value = best_value
        self.pso_optim = pso_optim
        self.current_fitness = fit

    def update_velocity_winner(self, upper_best, global_best, is_top=False):
        dim = self.velocity.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)

        if not is_top:
            random_coeff4 = np.random.rand(dim)
            updated_velocity = (random_coeff1 * self.velocity +
                                random_coeff2 * (upper_best - self.position) +
                                random_coeff3 * (self.best_position - self.position) +
                                self.pso_optim.phi * random_coeff4 * (global_best - self.position))
            updated_position = self.position + updated_velocity
        else:
            updated_velocity = self.velocity
            updated_position = self.position
        
        self.velocity = updated_velocity
        self.position = updated_position
        self.check()

    def update_velocity_loser(self, winner_pos):
        dim = self.velocity.shape[0]
        random_coeff1 = np.random.rand(dim)
        random_coeff2 = np.random.rand(dim)
        random_coeff3 = np.random.rand(dim)

        updated_velocity = (random_coeff1 * self.velocity +
                            random_coeff2 * (winner_pos - self.position) +
                            random_coeff3 * (self.best_position - self.position))
        updated_position = self.position + updated_velocity
        
        self.velocity = updated_velocity
        self.position = updated_position
        self.check()

    def check(self):
        self.velocity = self.pso_optim.constrain_velocity(self.velocity)
        self.position = self.pso_optim.constrain(self.position)
        self.fitness_check()

    def fitness_check(self):
        fit_res = self.pso_optim.fitness(self.position)
        value = self.pso_optim.unpack_fitness(fit_res)
        self.current_fitness = value
        if value < self.best_value:
            self.best_position = np.copy(self.position)
            self.best_value = value

class PPSO_optim:
    def __init__(self, config):
        self.layer_cfg = config['layer_config']
        self.phi = config['phi']
        self.particle_num = np.sum(self.layer_cfg)
        self.max_fes = config['max_fes']
        self.dimensions = config['dimensions']
        self.iterations = config['iterations']
        self.lb = config['lb']
        self.ub = config['ub']
        self.func_id = config['func_id']
        self.particle_vals = self.init_particles()
        self.best_solution = None

    def init_particles(self):
        particle_vals = []
        for _ in range(self.particle_num):
            position = self.lb + (self.ub - self.lb) * np.random.rand(self.dimensions)
            velocity = 0.1 * (self.lb + (self.ub - self.lb) * np.random.rand(self.dimensions))
            particle_vals.append((position, velocity, position, float('inf')))
        return particle_vals

    def sorted_particles(self, particles, reverse=False):
        coeff = 1 if reverse else -1
        sorted_indices = np.argsort(coeff * np.array([particle.current_fitness for particle in particles]))
        return np.array(particles)[sorted_indices]

    def fitness(self, position):
        if self.func_id == 1:
            return self.griewank(position)
        else:
            raise ValueError("Unsupported function ID")

    def griewank(self, x):
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_part - prod_part

    def constrain(self, position):
        return np.clip(position, self.lb, self.ub)

    def constrain_velocity(self, velocity):
        return velocity  # 根据需要调整速度的限制

    def unpack_fitness(self, fit_res):
        return fit_res  # 直接返回适应度值

    def set_best(self, value, position):
        if self.best_solution is None or value < self.best_solution['value']:
            self.best_solution = {'value': value, 'position': position}
            print(f"best value: {value}")

    def _algorithm(self):
        particles = [Particle_PPSO(*vals, self) for vals in self.particle_vals]
        global_best_particle = min(particles, key=lambda p: p.best_value)
        self.set_best(global_best_particle.best_value, global_best_particle.best_position)

        layers_num = len(self.layer_cfg)
        fes = 0

        for _ in range(self.iterations):

            particles = self.sorted_particles(particles, True)
            current_moved_particles = 0

            for layer_idx in range(layers_num - 1, -1, -1):
                layer_size = self.layer_cfg[layer_idx]
                if current_moved_particles == 0:
                    layer_particles = particles[-layer_size:]
                else:
                    start_idx = layer_size + current_moved_particles
                    layer_particles = particles[-start_idx:-current_moved_particles]
                current_moved_particles += layer_size

                rand_indices = np.random.permutation(layer_size)
                separator = layer_size // 2
                rand_pairs = np.column_stack((rand_indices[:separator], rand_indices[separator:2 * separator]))

                comparison_mask = [
                    layer_particles[rand_pairs[i, 0]].current_fitness > layer_particles[rand_pairs[i, 1]].current_fitness
                    for i in range(len(rand_pairs))
                ]

                loser_indices = np.where(comparison_mask, rand_pairs[:, 0], rand_pairs[:, 1])
                winner_indices = np.where(comparison_mask, rand_pairs[:, 1], rand_pairs[:, 0])
                losers = layer_particles[loser_indices]
                winners = layer_particles[winner_indices]

                top_layer_size = self.layer_cfg[0]
                top_layer_particles = particles[:top_layer_size]
                top_indices = np.random.permutation(separator) % top_layer_size
                aim_top_particles = top_layer_particles[top_indices]

                is_top_layer = layer_idx == 0

                if not is_top_layer:
                    upper_layer_size = self.layer_cfg[layer_idx - 1]
                    start_idx = -np.sum(self.layer_cfg[layer_idx-1:])
                    end_idx = start_idx + upper_layer_size
                    upper_layer_particles = particles[start_idx:end_idx]
                    upper_indices = np.random.permutation(separator) % upper_layer_size
                    aim_upper_particles = upper_layer_particles[upper_indices]

                for index in range(separator):
                    winner = winners[index]
                    loser = losers[index]

                    loser.update_velocity_loser(winner.position)
                    if is_top_layer:
                        winner.update_velocity_winner(None, None, is_top_layer)
                    else:
                        upper_best = aim_upper_particles[index]
                        global_best = aim_top_particles[index]
                        winner.update_velocity_winner(upper_best.position, global_best.position, is_top_layer)

                    self.set_best(winner.best_value, winner.best_position) if winner.best_value < loser.best_value else self.set_best(loser.best_value, loser.best_position)

                fes += len(particles)

        return self.best_solution
    
# 示例配置
config = {
    'layer_config': [4, 8, 20, 32],
    'phi': 0.02,
    'max_fes': 300000,
    'iterations': 4687,
    'lb': -600,
    'ub': 600,
    'dimensions' : 30,
    'func_id': 1
}
np.random.seed(74)
# 创建并运行PPSO优化器
ppso_optimizer = PPSO_optim(config)
best_solution = ppso_optimizer._algorithm()

print("Best solution:", best_solution)
