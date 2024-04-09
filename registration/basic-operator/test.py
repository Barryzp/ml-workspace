import itk
import numpy as np
from glob import glob
import os
import yaml
from pathlib import Path

import os
print(os.getcwd())


with open(Path("basic-operator/test.yml")) as f:
    yaml_config = yaml.safe_load(f)

yaml_config
print(yaml_config.cement)

