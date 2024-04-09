from registration_framework import Registration
from pso_optim import PSO_optim
from config import Config

config = Config()
registration = Registration(config)
pso = PSO_optim(config)
registration.set_optim_algorithm(pso)
registration.registrate()
