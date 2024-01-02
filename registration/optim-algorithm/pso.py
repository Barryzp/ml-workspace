import torch
import random

a = 1.3
b = 90
# (a,a^2有极小值)
def fintess(x):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

# 需要进行调参，惯性权重，个体最优系数，种群最有系数

# Particle class
class Particle:
    def __init__(self, x0):
        self.position = x0
        self.velocity = random.uniform(-1, 1)
        self.best_position = x0
        self.best_value = fintess(x0)

    def update_velocity(self, global_best_position):
        w = 0.1  # Inertia weight
        c1 = 1    # Cognitive (particle's best) weight
        c2 = 2    # Social (swarm's best) weight

        r1 = random.random()
        r2 = random.random()

        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def move(self):
        self.position += self.velocity
        value = fintess(self.position)

        if value < self.best_value:
            self.best_position = self.position
            self.best_value = value


# PSO algorithm
def algorithm(particle_vals, num_iterations):
    particles = [Particle(particle_vals[i]) for i in range(len(particle_vals))]
    global_best_position = min(particles, key=lambda p: p.best_value).position

    for _ in range(num_iterations):
        for particle in particles:
            particle.update_velocity(global_best_position)
            particle.move()

            global_best_val = fintess(global_best_position)
            print(global_best_val)
            if particle.best_value < global_best_val:
                global_best_position = particle.best_position

    return global_best_position

# 生成初始参数规定范围，
minV, maxV = -2.0, 2.0
size = [2]

particle_num = 100
iteratons = 50
poses = [torch.rand(size) * (maxV - minV) + minV for i in range(particle_num)]

# Running PSO
best_position = algorithm(poses, iteratons)
print(f"The best position found is: {best_position}")
print(f"The minimum value of the function is: {fintess(best_position)}")