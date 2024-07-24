import numpy as np

class Particle:
    def __init__(self, dim, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        self.velocity = np.random.uniform(-1, 1, dim)
        self.pbest_position = self.position.copy()
        self.pbest_value = float('inf')
        self.dim = dim

    def evaluate(self, objective_function):
        fitness = objective_function(self.position)
        if fitness < self.pbest_value:
            self.pbest_value = fitness
            self.pbest_position = self.position.copy()

    def update_velocity(self, gbest_position, w=0.5, c1=1, c2=2):
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        cognitive_velocity = c1 * r1 * (self.pbest_position - self.position)
        social_velocity = c2 * r2 * (gbest_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(self.dim):
            if self.position[i] > bounds[1]:
                self.position[i] = bounds[1]
            if self.position[i] < bounds[0]:
                self.position[i] = bounds[0]

def griewank_function(x):
    sum_sq = np.sum(x ** 2) / 4000
    cos_product = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_sq - cos_product

def pso(objective_function, dim, bounds, num_particles, max_iter):
    swarm = [Particle(dim, bounds) for _ in range(num_particles)]
    gbest_value = float('inf')
    gbest_position = np.random.uniform(bounds[0], bounds[1], dim)

    for iteration in range(max_iter):
        for particle in swarm:
            particle.update_velocity(gbest_position)
            particle.update_position(bounds)

            particle.evaluate(objective_function)
            if particle.pbest_value < gbest_value:
                gbest_value = particle.pbest_value
                gbest_position = particle.pbest_position.copy()

        print(f"Iteration {iteration+1}/{max_iter}, Global Best Value: {gbest_value}")

    return gbest_position, gbest_value

# PSO参数
dim = 30  # 问题的维度
bounds = (-600, 600)  # 搜索空间的边界
num_particles = 64  # 粒子数量
max_iter = 4687  # 最大迭代次数
np.random.seed(45)

# 运行PSO优化Griewank函数
best_position, best_value = pso(griewank_function, dim, bounds, num_particles, max_iter)
print("Best Position:", best_position)
print("Best Value:", best_value)
