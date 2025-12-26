"""
Bias - LLM Steering with Interpretable SAE Features
====================================================

Bias is a Python library for steering Large Language Models using
Sparse Autoencoder (SAE) features from Neuronpedia.

Quick Start
-----------
>>> from bias import Bias
>>> 
>>> bias = Bias("gpt2")
>>> bias.steer("professional formal writing")
>>> output = bias.generate("Write an email:")
>>> print(output)

For more information, see: https://bias.readthedocs.io
"""

__version__ = "0.1.0"
__author__ = "codewithdark-git"
__email__ = "codewithdark90@gmail.com"

# High-level API imports
from bias.core.config import NeuronpediaConfig, ModelConfig
from bias.core.client import NeuronpediaClient
from bias.core.engine import SteeringEngine
from bias.core.library import ConceptLibrary

# Convenience imports
from bias.api import Bias, steer, generate, discover_features

__all__ = [
    # Main high-level API
    "Bias",
    "steer",
    "generate",
    "discover_features",
    # Core components
    "SteeringEngine",
    "NeuronpediaClient",
    "NeuronpediaConfig",
    "ModelConfig",
    "ConceptLibrary",
    # Version
    "__version__",
]
