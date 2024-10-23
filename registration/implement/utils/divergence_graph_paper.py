import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DivergenceTools:

    def __init__(self, config) -> None:
        self.config = config

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
            # 4_0_PPSO_optim4_params_matched
            file_path = f"{csv_path}/{record_id}/{i}_0_{optim_method}_params_matched.csv"
            csv_datas = pd.read_csv(file_path)
            fes = csv_datas['iterations'].values
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
        total_data_rows = 0

        fun_fit_sort = {}
        for fun_id in fun_ids:
            fun_fit_sort.setdefault(fun_id, [])
        # 进行排名，排名之后进行平均
        optim_rank = {}
        for method_name in methods_name:
            optim_rank.setdefault(method_name, {})
            item1 = {}
            for fun_id in fun_ids:
                item2 = self._read_optims_data(method_name, record_id, fun_id, run_times)
                item1.setdefault(fun_id, item2)
                sort_item = {}
                sort_item.setdefault("class", method_name)
                sort_item.setdefault("fitness", item2["mean_best"])
                total_data_rows = len(item2["fes"])
                fun_fit_sort.get(fun_id).append(sort_item)
            data_dict.setdefault(method_name, item1)
    
        for fun_id in fun_ids:
            sort_arr = fun_fit_sort.get(fun_id)
            sort_arr = sorted(sort_arr, key=lambda x: x["fitness"])
            for i in range(len(sort_arr)):
                rank = i+1
                sort_item = sort_arr[i]
                optim_rank.get(sort_item["class"]).setdefault(fun_id, rank)

        for cls, ranks in optim_rank.items():
            mean_rank = 0
            fun_num = 0
            for fun_id, rank in ranks.items():
                mean_rank += rank
                fun_num += 1
            mean_rank = mean_rank / fun_num
            ranks.setdefault('mean_rank', mean_rank)
            print(f"{cls}, {ranks}")

        data_dict.setdefault("total_data_rows", total_data_rows)
        return data_dict

    # 展示收敛曲线，这个total_mark代表的是显示多少个mark，在折线上（展示的是中位数）
    def show_median_convergence_line(self, data_dict, methods_name, fun_id, total_mark = None):
        step = 1
        
        # 需要获取总的列数
        total_iters = data_dict["total_data_rows"]
        if total_mark != None:
            step = total_iters // total_mark

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
        
        # 需要获取总的列数
        total_iters = data_dict["total_data_rows"]
        if total_mark != None:
            step = total_iters // total_mark

        j = 0
        for method_name in methods_name:
            optim_item = data_dict[method_name]
            data_item = optim_item[fun_id]
            mean = data_item["mean_fes"]
            mean = mean[::step] - best_fit #np.log10(mean[::step] - best_fit)
            fes = data_item["fes"]
            fes = fes[::step]
            plt.plot(fes, mean, label=method_name,
                     color=self.config.colors[j], marker=self.config.markers[j])
            j+=1
        plt.xlabel('FEs')
        # plt.ylabel('f(x)-f(x*)log10')
        plt.legend()
        # 显示图像
        plt.show()