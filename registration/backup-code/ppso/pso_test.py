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

def shifted_sphere(x, o):
    """Shifted Sphere 函数"""
    return np.sum((x - o)**2)

def shifted_sphere_1(x):
    """Shifted Sphere 函数"""
    return np.sum(x**2)

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
bounds = (-100, 100)  # 搜索空间的边界
num_particles = 64  # 粒子数量
max_iter = 4687  # 最大迭代次数
np.random.seed(45)

# 生成偏移向量o
o = np.array([-21.98480969,  11.55499693, -36.01068093,  69.37273235,
       -37.60887075, -48.53629215,  53.7647669 ,  13.71856864,
        69.82858747, -18.62781124,  29.30660868, -70.21691829,
       -51.7402846 ,  71.73758557, -57.09778849,  74.86839208,
         7.55890615,  60.3877141 ,  15.72331187,  31.66238351,
       -49.3407677 ,  55.03788271, -52.66414674, -26.05238299,
        54.04788928, -77.47142157,  64.60508511, -17.71212496,
       -11.57427923, -42.59138623])

def f1(x):
    return shifted_sphere(x, o)


lb = np.full((dim), bounds[0])
ub = np.full((dim), bounds[1])
# %% Do PSO
from sko.PSO import PSO

pso = PSO(func=f1, n_dim=dim, pop=num_particles, max_iter=max_iter, lb=lb, ub=ub, w=0.5, c1=1, c2=2)
pso.run()

pso.gbest_x

import matplotlib.pyplot as plt
plt.xlabel("iterators", size=11)
plt.ylabel("fitness", size=11)
plt.plot(pso.gbest_y_hist)
# plt.plot(pso.gbest_y_hist, color='b', linewidth=2)
plt.show()

# 运行PSO优化Griewank函数
# best_position, best_value = pso(f1, dim, bounds, num_particles, max_iter)
# print("Best Position:", best_position)
# print("Best Value:", best_value)
