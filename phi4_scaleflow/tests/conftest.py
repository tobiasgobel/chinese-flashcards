import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
