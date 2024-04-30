import torch, math, random, cma
import numpy as np
from utils.tools import Tools
from optim_base import OptimBase

class CMAES(OptimBase):

    def __init__(self, config, share_records_out = None) -> None:
        super(CMAES, self).__init__(config, share_records_out)
        
        # CMA-ES的特殊初始化，正常情况下是个负值
        self.reset()

    def reset(self):
        self.best_value = 1000

    # 基本参数的初始化
    def init_basic_params(self):
        super(CMAES, self).init_basic_params()

        self.sigma0 = self.config.sigma0  # sigma0 初始高斯核标准差
        self.pop_size = self.config.pop_size    # pop_size 总群大小

    # 生成初始解
    def spawn_initial_guess(self):
        init_translate = self.config.init_translate
        translate_delta = self.config.translate_delta
        init_solve = [init_translate[0]+translate_delta[0]/2, init_translate[1]+translate_delta[1]/2]

        initial_guess = np.array([init_solve[0], init_solve[1], self.maxV[-1] / 2])
        return initial_guess

    def set_global_best_datas(self):
        super(CMAES, self).set_global_best_datas(-self.best_value, self.best_solution, self.best_match_img)

    # 定义目标函数，主义CMA-ES优化目标是最小值，我们想要求得最大值需要取个负号
    def objective_function(self, x):
        x = torch.tensor(x)
        res = self.fitness(x)
        fitness = -res[0]
        if fitness < self.best_value:
            self.best_value = fitness
            self.best_solution = x
            self.best_match_img = res[1]
            self.best_result_per_iter = res
            self.set_global_best_datas()
        return fitness

    # PSO algorithm
    def _algorithm(self):
        self.reset()
        bounds = np.array((self.minV, self.maxV))
         # 初始猜测解，根据问题维度调整
        initial_guess = self.spawn_initial_guess()

        sigma0 = self.config.sigma0
        pop_size = self.config.pop_size
        max_iter = self.config.max_iter

        # 初始化 CMA-ES 优化器
        es = cma.CMAEvolutionStrategy(
            initial_guess, 
            sigma0, 
            {   'popsize': pop_size, 
                'bounds': [bounds[0], bounds[1]],
                'maxiter': max_iter,
                #'verb_disp': 1,    # 每代打印一次信息
                })

        iterations = 0

        # 开始优化过程
        while not es.stop():
            # 获取一组新的样本点
            solutions = es.ask()
            # 计算每个样本点的目标函数值
            function_values = [self.objective_function(solution) for solution in solutions]
            # 每一次迭代保存一下
            self.save_iter_records(iterations)

            check = self.check_match_finished()
            if check : return

            print(f"iterations solution found: {self.best_solution}")
            print(f"iterations fitness achieved: {-self.best_value}")

            # 将目标函数值反馈给 CMA-ES
            es.tell(solutions, function_values)
            # 输出当前的状态信息（可选）
            es.logger.add()  # write data to logger
            iterations+=1

        # 获取最终的最佳解
        best_solution = es.result.xbest
        best_fitness = es.result.fbest
        # print("CMA-ES run finished!")

    # 进行优化
    def run(self):
        # Running CMA-ES
        self._algorithm()
        best_val = -self.best_value
        print(f"The best position found is: {self.best_solution}")
        print(f"The maximum value of the function is: {best_val}")
        self.put_best_data_in_share(best_val, self.best_solution)
        self.save_iteration_params()
        if self.config.mode == "matched":
            # 保存相关数据(图像之类的)
            self.save_iteration_best_reg_img(self.best_match_img, self.config.max_iter)

        print("===================================divided line.=================================\n")
        return best_val, self.best_match_img