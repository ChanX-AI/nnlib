"""
NNLib: A Simple Neural Network Library
----------------------------------------
A lightweight neural network framework build from scratch.

Author: Chandu
Version: 0.1.0
Purpose: Educational and experimental use
"""



from . import layers
from . import losses
from . import model
from . import optims
from . import utils

# version
__version__ = '0.1.0'

__all__ = ['layers', 'losses', 'model', 'optims', 'utils']