import numpy as np

from registration_framework import Registration
from optims.cmaes_optim import CMAES
from optims.cso_optim import CSO_optim
from optims.pso_optim import PSO_optim
from optims.ppso_optim import PPSO_optim
from optims.ppso_optim_improved1 import PPSO_optim1, PPSO_optim2, PPSO_optim3, PPSO_optim1_1
from utils.tools import Tools
from optim_test_fun import OptimFunTest
import os
config = Tools.load_yaml_config("implement/configs/optim_test_config.yaml")
res_path = f"{config.data_save_path}/{config.record_id}"
file_name = f"{config.record_id}_config.yaml"
Tools.save_obj_yaml(res_path, file_name, config)
config.max_fes = config.solution_dimension * 10000

run_times = config.run_times
# 运行测试函数
test_fun_framework = OptimFunTest(config)
optim_classes = [PSO_optim, PPSO_optim, PPSO_optim1, PPSO_optim1_1, PPSO_optim2, PPSO_optim3, CSO_optim]
test_fun_framework.test_all_optims_multi_process(optim_classes)