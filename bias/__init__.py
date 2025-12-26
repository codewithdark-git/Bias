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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bias.core.config import BiasConfig, NeuronpediaConfig, ModelConfig, configure
    from bias.core.client import NeuronpediaClient
    from bias.core.engine import SteeringEngine
    from bias.core.library import ConceptLibrary
    from bias.api import Bias

# Lazy imports to avoid torch dependency during installation
def __getattr__(name):
    if name in ("BiasConfig", "NeuronpediaConfig", "ModelConfig", "configure"):
        from bias.core.config import BiasConfig, NeuronpediaConfig, ModelConfig, configure
        return locals()[name] if name in locals() else globals()[name]
    elif name in ("NeuronpediaClient", "SteeringEngine", "ConceptLibrary"):
        from bias.core import NeuronpediaClient, SteeringEngine, ConceptLibrary
        return locals()[name] if name in locals() else globals()[name]
    elif name in ("Bias", "steer", "generate", "discover_features"):
        from bias.api import Bias, steer, generate, discover_features
        return locals()[name] if name in locals() else globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Main high-level API
    "Bias",
    "BiasConfig",
    "configure",
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
