import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cec2013.cec2013 import CEC_functions

class Test_Fun:
    # 非并行化的方法，单个粒子，原测试函数是一个张量进行计算的
    # 此外这个基准函数目标是最小值，这个地方需要改一改，基准函数没问题
    def griewank(x):
        sum_sq = np.sum(x ** 2) / 4000
        cos_product = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq - cos_product
    
    def generate_random_rotation_matrix(dim):
        """生成随机正交矩阵"""
        random_matrix = np.random.randn(dim, dim)
        q, _ = np.linalg.qr(random_matrix)
        return q

    def weierstrass(x, a=0.5, b=3, k_max=20):
        """原始Weierstrass函数"""
        n = len(x)
        sum1 = 0
        sum2 = 0
        for i in range(n):
            sum1 += sum(a**k * np.cos(2 * np.pi * b**k * (x[i] + 0.5)) for k in range(k_max + 1))
        sum2 = n * sum(a**k * np.cos(2 * np.pi * b**k * 0.5) for k in range(k_max + 1))
        return sum1 - sum2

    def rotated_weierstrass(x, rotation_matrix):
        """旋转后的Weierstrass函数"""
        rotated_x = np.dot(rotation_matrix, x)
        return Test_Fun.weierstrass(rotated_x)

    def sphere(x):
        """Sphere函数"""
        return np.sum(x**2)

class OptimFunTest:

    def __init__(self, config) -> None:
        self.config = config
        self.cec2013 = CEC_functions(config.solution_dimension)

    def custom_fitness_obj(self, optim, fun_id):
        rotation_mat = None
        dim = self.config.solution_dimension
        if fun_id == "rotated_weierstrass":
            rotation_mat = Test_Fun.generate_random_rotation_matrix(dim)

        fun = getattr(Test_Fun, fun_id)

        fitness_coeff = 1 if self.config.target_max else -1

        # 目标是最小化函数
        def fitness_fun(x):
            if rotation_mat is None : return fun(x) * fitness_coeff 
            return fun(x, rotation_mat) * fitness_coeff

        optim.set_init_params(fitness_fun, self, fun_id)

    def cec2013_fitness(self, optim, fun_id):
        fitness_coeff = 1 if self.config.target_max else -1
        best_fitness = self.config.fun_configs["cec2013"][fun_id]["best_fit"]

        def fitness_fun(x):
            return fitness_coeff * (self.cec2013.Y(x, fun_id) - best_fitness)
        optim.set_init_params(fitness_fun, self, fun_id)

    # 设置适应值评估函数，根据optim的配置中的funid来
    def set_fitness_obj(self, optim, fun_id):
        if self.config.test_type == "normal":
            return self.custom_fitness_obj(optim, fun_id)
        elif self.config.test_type == "cec2013":
            return self.cec2013_fitness(optim, fun_id)

    # 默认测试单一优化算法，目标函数对应config中的funid, random为true，不固定随机种子
    def test_single_optim_fun_id(self, optim_class, random = True):
        run_id = -1
        fun_id = self.config.fun_id
        rand_seed = self.config.rand_seed
        # 每一次迭代随机种子+1，这样的方式保证结果的一致
        if not random : np.random.seed(rand_seed)
        optim = optim_class(self.config)
        optim.set_runid(run_id)
        self.set_fitness_obj(optim, fun_id)
        optim.run_std_optim()

    # 测试测试函数和优化算法
    def test_single_optim_fun(self, optim_class, fun_id):
        run_times = self.config.run_times

        for i in range(run_times):
            rand_seed = self.config.rand_seed + i
            # 每一次迭代随机种子+1，这样的方式保证结果的一致
            np.random.seed(rand_seed)
            optim = optim_class(self.config)
            optim.set_runid(i)
            self.set_fitness_obj(optim, fun_id)
            optim.run_std_optim()

    # 测试优化算法的一系列的测试函数
    def test_optim_funs(self, optim_class):
        fun_ids = self.config.fun_ids
        for fun_id in fun_ids:
            self.test_single_optim_fun(optim_class, fun_id)

    # 测试所有的优化函数和优化算法
    def test_all_optims_funs(self, optim_classes):
        for optim in optim_classes:
            self.test_optim_funs(optim)

    # 读取优化算法对应的相关数据
    # optim_method：对应优化算法
    # record_id：保存的文件夹名
    # fun_id: 测试函数名
    # run_times：运行了多少次独立实验
    def _read_optims_data(self, optim_method, record_id, fun_id, run_times):
        # 先构造dict
        csv_path = self.config.data_save_path
        record_id = self.config.record_id
        data_item = []
        for i in range(run_times):
            file_path = f"{csv_path}/{record_id}/{fun_id}/{optim_method}_fes_{i}.csv"
            csv_datas = pd.read_csv(file_path)
            fes = csv_datas['FEs'].values
            fitness = csv_datas['fitness'].values
            data_item.append(fitness)
        
        # shape: (run_times, iter_fitness)
        np_arr = np.stack(data_item)
        iter_best = np_arr[:, -1]
        mean_best_fit = np.mean(iter_best)
        std_best_fit = np.std(iter_best)
        med = np.median(np_arr, axis=0)
        mean_fes = np.mean(np_arr, axis=0)
        data_dict = {
            "mean_best" : mean_best_fit, 
            "std_best_fit" : std_best_fit, 
            "median_fes" : med,
            "mean_fes" : mean_fes, 
            "fes" : fes
        }
        if self.config.show_log:
            print(f"{optim_method}, {fun_id}, mean: {mean_best_fit:.4e}, std: {std_best_fit:.4e}")
        return data_dict

    def read_optims_fitness_FEs(self, methods_name):
        data_dict = {}
        record_id = self.config.record_id
        fun_ids = self.config.fun_ids
        run_times = self.config.run_times
        for method_name in methods_name:
            item1 = {}
            for fun_id in fun_ids:
                item2 = self._read_optims_data(method_name, record_id, fun_id, run_times)
                item1.setdefault(fun_id, item2)
            data_dict.setdefault(method_name, item1)
        return data_dict

    # 展示收敛曲线，这个total_mark代表的是显示多少个mark，在折线上（展示的是中位数）
    def show_median_convergence_line(self, data_dict, methods_name, fun_id, total_mark = None):
        step = 1
        
        total_fes = self.config.iteratons
        if total_mark != None:
            step = total_fes // total_mark

        j = 0
        for method_name in methods_name:
            optim_item = data_dict[method_name]
            data_item = optim_item[fun_id]
            median = data_item["median_fes"]
            median = median[::step]
            fes = data_item["fes"]
            fes = fes[::step]
            plt.plot(fes, median, label=method_name,
                     color=self.config.colors[j], marker=self.config.markers[j])
            j+=1
        plt.xlabel('FEs')
        plt.ylabel('f(x)-f(x*)')
        plt.legend()
        # 显示图像
        plt.show()

    # 展示收敛曲线，这个total_mark代表的是显示多少个mark，在折线上（展示的是均值），纵轴上是以log10的对数
    def show_mean_convergence_line(self, data_dict, methods_name, fun_id, total_mark = None):
            step = 1

            best_fit = 0.0
            if self.config.test_type == "normal":
                fun_config = self.config.fun_configs[fun_id]
                best_fit = fun_config["fun_config"]
            elif self.config.test_type == "cec2013":
                fun_config = self.config.fun_configs["cec2013"][fun_id]
            total_fes = self.config.iteratons
            if total_mark != None:
                step = total_fes // total_mark

            j = 0
            for method_name in methods_name:
                optim_item = data_dict[method_name]
                data_item = optim_item[fun_id]
                mean = data_item["mean_fes"]
                mean = np.log10(mean[::step] - best_fit)
                fes = data_item["fes"]
                fes = fes[::step]
                plt.plot(fes, mean, label=method_name,
                         color=self.config.colors[j], marker=self.config.markers[j])
                j+=1
            plt.xlabel('FEs')
            plt.ylabel('f(x)-f(x*)log10')
            plt.legend()
            # 显示图像
            plt.show()