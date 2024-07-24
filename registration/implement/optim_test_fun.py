import numpy as np


class OptimFunTest:

    def __init__(self, config) -> None:
        self.config = config


    # 设置适应值评估函数，根据optim的配置中的funid来
    def set_fitness_obj(self, optim):
        fun = getattr(OptimFunTest, optim.config.fun_id)

        fitness_coeff = 1 if self.config.target_max else -1

        # 目标是最小化函数
        def fitness_fun(x):
            return fun(x) * fitness_coeff

        optim.set_init_params(fitness_fun, self)

    # 非并行化的方法，单个粒子，原测试函数是一个张量进行计算的
    # 此外这个基准函数目标是最小值，这个地方需要改一改，基准函数没问题
    def griewank(x):
        sum_sq = np.sum(x ** 2) / 4000
        cos_product = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq - cos_product
