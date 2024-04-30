from registration_framework import Registration
from pso_optim import PSO_optim
from utils.tools import Tools

config = Tools.load_yaml_config("config.yaml")
res_path = f"{config.data_save_path}/{config.record_id}"
file_name = f"{config.record_id}_config.yaml"
Tools.save_obj_yaml(res_path, file_name, config)

registration = Registration(config)
pso = PSO_optim(config)
registration.set_optim_algorithm(pso)
result = registration.registrate()
