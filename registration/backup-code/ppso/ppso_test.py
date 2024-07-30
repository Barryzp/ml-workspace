import numpy as np

class Particle:
    def __init__(self, position, velocity, fitness):
        self.position = position
        self.velocity = velocity
        self.personal_best_position = position.copy()
        self.fitness = fitness
        self.personal_best_fitness = fitness

class PSO:
    def __init__(self, layers, dimensions, lower_bound, upper_bound, max_fes, o,func_id):
        self.layers = layers
        self.dimensions = dimensions
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_fes = max_fes
        self.func_id = func_id
        self.o = o

        self.num_layers = len(layers)
        self.cumulative_layers = np.cumsum(layers)
        self.total_particles = np.sum(layers)
        self.fes = 0

        if dimensions == 50:
            self.phi = 0.04
        elif dimensions == 30:
            self.phi = 0.02
        else:
            self.phi = 0.008

        self.position_min = np.tile(lower_bound, (self.total_particles, 1))
        self.position_max = np.tile(upper_bound, (self.total_particles, 1))
        self.particles = []

        for _ in range(self.total_particles):
            position = self.position_min[0] + (self.position_max[0] - self.position_min[0]) * np.random.rand(dimensions)
            velocity = 0.1 * (self.position_min[0] + (self.position_max[0] - self.position_min[0]) * np.random.rand(dimensions))
            fitness = self.evaluate(position)
            self.particles.append(Particle(position, velocity, fitness))

        self.global_best_particle = min(self.particles, key=lambda p: p.fitness)
        self.best_ever_fitness = self.global_best_particle.fitness
        self.best_fitness_history = []
        self.max_generations = self.max_fes // self.total_particles

    def evaluate(self, position):
        if self.func_id == 1:
            return self.shifted_sphere(position, self.o)
        else:
            raise ValueError("Unsupported func_id")

    @staticmethod
    def griewank(x):
        sum_term = np.sum(x ** 2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[0] + 1))))
        return sum_term - prod_term + 1
    
    @staticmethod
    def shifted_sphere(x, o):
        """Shifted Sphere 函数"""
        return np.sum((x - o)**2)

    def optimize(self):
        generation = 0
        while self.fes < self.max_fes:
            self.particles.sort(key=lambda p: p.fitness)
            self.best_fitness_history.append(self.particles[0].fitness)
            print(f"generation: {generation} best value: {self.particles[0].fitness}")
            particle_groups = [[] for _ in range(self.num_layers)]
            for i in range(self.num_layers):
                if i == 0:
                    indices = range(self.cumulative_layers[0])
                else:
                    indices = range(self.cumulative_layers[i - 1], self.cumulative_layers[i])
                for j in indices:
                    particle_groups[i].append(self.particles[j])

            for i in range(self.num_layers - 1, -1, -1):
                layer_particles = particle_groups[i]
                random_indices = np.random.permutation(len(layer_particles))
                separator = len(layer_particles) // 2
                random_pairs = [(random_indices[j], random_indices[j + separator]) for j in range(separator)]

                for loser_idx, winner_idx in random_pairs:
                    loser = layer_particles[loser_idx]
                    winner = layer_particles[winner_idx]

                    if loser.fitness > winner.fitness:
                        loser, winner = winner, loser

                    rand_co1 = np.random.rand(self.dimensions)
                    rand_co2 = np.random.rand(self.dimensions)
                    rand_co3 = np.random.rand(self.dimensions)

                    loser.velocity = (rand_co1 * loser.velocity +
                                      rand_co2 * (winner.position - loser.position) +
                                      rand_co3 * (loser.personal_best_position - loser.position))
                    loser.position += loser.velocity
                    loser.position = np.clip(loser.position, self.lower_bound, self.upper_bound)
                    loser.fitness = self.evaluate(loser.position)

                    if loser.fitness < loser.personal_best_fitness:
                        loser.personal_best_position = loser.position.copy()
                        loser.personal_best_fitness = loser.fitness

                    if i != 0:
                        upper_layer_particles = particle_groups[i - 1]
                        random_upper_idx = np.random.choice(len(upper_layer_particles))
                        upper_position = upper_layer_particles[random_upper_idx].position

                        rand_co4 = np.random.rand(self.dimensions)
                        winner.velocity = (rand_co1 * winner.velocity +
                                           rand_co2 * (upper_position - winner.position) +
                                           rand_co3 * (winner.personal_best_position - winner.position) +
                                           self.phi * rand_co4 * (self.global_best_particle.position - winner.position))
                        winner.position += winner.velocity
                        winner.position = np.clip(winner.position, self.lower_bound, self.upper_bound)
                        winner.fitness = self.evaluate(winner.position)

                        if winner.fitness < winner.personal_best_fitness:
                            winner.personal_best_position = winner.position.copy()
                            winner.personal_best_fitness = winner.fitness

            self.particles = [p for layer in particle_groups for p in layer]
            self.global_best_particle = min(self.particles, key=lambda p: p.fitness)
            self.best_ever_fitness = min(self.best_ever_fitness, self.global_best_particle.fitness)
            self.fes += self.total_particles
            generation += 1

        return self.best_ever_fitness, [p.fitness for p in self.particles], self.particles, self.best_fitness_history


def main():
    np.random.seed(45)
    runs = 10
    layers = [4, 8, 20, 32]
    dimensions = 30
    max_fes = dimensions * 10000
    func_id = 1
    lower_bound = -600
    upper_bound = 600
    results = np.ones(runs) * 99999999999
    # 生成偏移向量o
    o = np.random.uniform(lower_bound, upper_bound, dimensions)
    for run_index in range(runs):
        pso = PSO(layers, dimensions, lower_bound, upper_bound, max_fes, o, func_id)
        best_fitness, fitness, particles, best_fitness_history = pso.optimize()
        results[run_index] = best_fitness
        print(f'{run_index + 1} : {results[run_index]:e}')

    print('\n\n====================\n\n')
    print(f'FID:{func_id} mean result: {np.mean(results):e}')


if __name__ == "__main__":
    main()
