"""
Configuration Classes for Bias
==============================

This module provides configuration dataclasses for setting up
the Bias steering system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class NeuronpediaConfig:
    """
    Configuration for Neuronpedia API access.
    
    Neuronpedia provides pre-trained Sparse Autoencoders (SAEs) with
    interpretable features for various language models.
    
    Attributes
    ----------
    api_key : str, optional
        API key for authenticated access (higher rate limits)
    base_url : str
        Base URL for Neuronpedia API
    model_id : str
        Model identifier in Neuronpedia (e.g., "gpt2-small", "llama-7b")
    layer : int
        Which transformer layer's SAE to use for steering
    sae_id : str
        SAE variant identifier (e.g., "res-jb" for residual stream)
    
    Examples
    --------
    >>> config = NeuronpediaConfig(
    ...     model_id="gpt2-small",
    ...     layer=6,
    ...     sae_id="res-jb"
    ... )
    
    Notes
    -----
    Different layers capture different levels of abstraction:
    - Early layers (0-4): Token-level, syntactic features
    - Middle layers (5-8): Semantic, conceptual features  
    - Later layers (9+): Task-specific, output-focused features
    """
    api_key: Optional[str] = None
    base_url: str = "https://neuronpedia.org/api"
    model_id: str = "gpt2-small"
    layer: int = 6
    sae_id: str = "res-jb"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_id": self.model_id,
            "layer": self.layer,
            "sae_id": self.sae_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuronpediaConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """
    Configuration for the target language model.
    
    Attributes
    ----------
    model_name : str
        HuggingFace model name or path
    device : str
        Device to load model on ("auto", "cuda", "cpu", "mps")
    dtype : str
        Model precision ("float16", "bfloat16", "float32")
    load_in_8bit : bool
        Whether to use 8-bit quantization
    load_in_4bit : bool
        Whether to use 4-bit quantization
    trust_remote_code : bool
        Whether to trust remote code in model loading
    
    Examples
    --------
    >>> config = ModelConfig(
    ...     model_name="meta-llama/Llama-2-7b-hf",
    ...     device="auto",
    ...     dtype="float16"
    ... )
    """
    model_name: str = "gpt2"
    device: str = "auto"
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": self.dtype,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass 
class SteeringConfig:
    """
    Configuration for steering behavior.
    
    Attributes
    ----------
    default_intensity : float
        Default steering intensity (1.0 = normal, higher = stronger)
    num_features : int
        Number of features to use per concept
    auto_balance : bool
        Whether to weight features by relevance score
    apply_to_layers : list
        Which layers to apply steering (empty = use neuronpedia config layer)
    normalize_vectors : bool
        Whether to normalize steering vectors
    """
    default_intensity: float = 1.0
    num_features: int = 5
    auto_balance: bool = True
    apply_to_layers: list = field(default_factory=list)
    normalize_vectors: bool = True


# Pre-configured model mappings
SUPPORTED_MODELS = {
    "gpt2": {
        "hf_name": "gpt2",
        "neuronpedia_id": "gpt2-small",
        "recommended_layer": 6,
        "sae_id": "res-jb",
    },
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "neuronpedia_id": "gpt2-medium", 
        "recommended_layer": 12,
        "sae_id": "res-jb",
    },
    "gpt2-large": {
        "hf_name": "gpt2-large",
        "neuronpedia_id": "gpt2-large",
        "recommended_layer": 18,
        "sae_id": "res-jb",
    },
    "gpt2-xl": {
        "hf_name": "gpt2-xl",
        "neuronpedia_id": "gpt2-xl",
        "recommended_layer": 24,
        "sae_id": "res-jb",
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get recommended configuration for a supported model.
    
    Parameters
    ----------
    model_name : str
        Model name (e.g., "gpt2", "gpt2-medium")
    
    Returns
    -------
    dict
        Configuration dictionary with hf_name, neuronpedia_id, etc.
    
    Raises
    ------
    ValueError
        If model is not in the supported models list
    """
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    raise ValueError(
        f"Model '{model_name}' not in supported models. "
        f"Supported: {list(SUPPORTED_MODELS.keys())}"
    )

