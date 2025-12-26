"""
Configuration Classes for Bias
==============================

This module provides configuration dataclasses for setting up
the Bias steering system.

Quick Start
-----------
>>> from bias import BiasConfig, Bias
>>> 
>>> # Create configuration with your API key
>>> config = BiasConfig(
...     api_key="your-neuronpedia-api-key",
...     model="gpt2",
... )
>>> 
>>> # Pass config to Bias
>>> bias = Bias(config=config)
>>> bias.steer("professional writing")
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union


# Pre-configured model mappings (must be defined before BiasConfig)
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


@dataclass
class BiasConfig:
    """
    Main configuration class for the Bias library.
    
    This is the recommended way to configure Bias. Create a config object
    with your settings and pass it to the Bias class.
    
    Parameters
    ----------
    api_key : str, optional
        Your Neuronpedia API key for higher rate limits.
        Can also be set via NEURONPEDIA_API_KEY environment variable.
    model : str, default="gpt2"
        Model to use. Supported: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
        Or any HuggingFace model ID.
    layer : int, optional
        Which layer to apply steering to. If not specified,
        uses the recommended layer for the model.
    sae_id : str, default="res-jb"
        SAE variant identifier (e.g., "res-jb" for residual stream)
    device : str, default="auto"
        Device to load model on ("auto", "cuda", "cpu", "mps")
    dtype : str, default="float16"
        Model precision ("float16", "bfloat16", "float32")
    intensity : float, default=1.0
        Default steering intensity (1.0 = normal, higher = stronger)
    num_features : int, default=5
        Default number of features to use per concept
    library_path : str, default="bias_concepts.json"
        Path to save/load concept library
    
    Examples
    --------
    Basic usage with API key:
    
    >>> config = BiasConfig(api_key="your-api-key")
    >>> bias = Bias(config=config)
    
    Using environment variable for API key:
    
    >>> # Set NEURONPEDIA_API_KEY in your environment
    >>> config = BiasConfig(model="gpt2-medium")
    >>> bias = Bias(config=config)
    
    Full configuration:
    
    >>> config = BiasConfig(
    ...     api_key="your-api-key",
    ...     model="gpt2-large",
    ...     layer=18,
    ...     device="cuda",
    ...     intensity=2.0,
    ... )
    >>> bias = Bias(config=config)
    
    Notes
    -----
    API Key Priority:
    1. Explicitly passed api_key parameter
    2. NEURONPEDIA_API_KEY environment variable
    3. None (uses unauthenticated access with lower rate limits)
    """
    # API Configuration
    api_key: Optional[str] = None
    base_url: str = "https://www.neuronpedia.org"
    
    # Model Configuration
    model: str = "gpt2"
    layer: Optional[int] = None
    sae_id: str = "res-jb"
    
    # Device Configuration
    device: str = "auto"
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    
    # Steering Configuration
    intensity: float = 1.0
    num_features: int = 5
    normalize_vectors: bool = True
    
    # Library Configuration
    library_path: str = "bias_concepts.json"
    
    def __post_init__(self):
        """Initialize configuration after dataclass creation."""
        # Check for API key in environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("NEURONPEDIA_API_KEY")
        
        # Resolve model configuration
        if self.model in SUPPORTED_MODELS:
            model_info = SUPPORTED_MODELS[self.model]
            self._hf_model = model_info["hf_name"]
            self._neuronpedia_id = model_info["neuronpedia_id"]
            self._default_layer = model_info["recommended_layer"]
            if self.sae_id == "res-jb":  # Only override if using default
                self.sae_id = model_info["sae_id"]
        else:
            self._hf_model = self.model
            self._neuronpedia_id = self.model
            self._default_layer = 6
        
        # Set layer to default if not specified
        if self.layer is None:
            self.layer = self._default_layer
    
    @property
    def hf_model_name(self) -> str:
        """Get the HuggingFace model name."""
        return self._hf_model
    
    @property
    def neuronpedia_model_id(self) -> str:
        """Get the Neuronpedia model ID."""
        return self._neuronpedia_id
    
    def to_neuronpedia_config(self) -> "NeuronpediaConfig":
        """Convert to NeuronpediaConfig for internal use."""
        return NeuronpediaConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model_id=self._neuronpedia_id,
            layer=self.layer,
            sae_id=self.sae_id,
        )
    
    def to_model_config(self) -> "ModelConfig":
        """Convert to ModelConfig for internal use."""
        return ModelConfig(
            model_name=self._hf_model,
            device=self.device,
            dtype=self.dtype,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            trust_remote_code=self.trust_remote_code,
        )
    
    def to_steering_config(self) -> "SteeringConfig":
        """Convert to SteeringConfig for internal use."""
        return SteeringConfig(
            default_intensity=self.intensity,
            num_features=self.num_features,
            normalize_vectors=self.normalize_vectors,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "layer": self.layer,
            "sae_id": self.sae_id,
            "device": self.device,
            "dtype": self.dtype,
            "intensity": self.intensity,
            "num_features": self.num_features,
            "library_path": self.library_path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BiasConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_env(cls, **kwargs) -> "BiasConfig":
        """
        Create config from environment variables.
        
        Environment Variables
        ---------------------
        NEURONPEDIA_API_KEY : API key for Neuronpedia
        BIAS_MODEL : Model name (default: gpt2)
        BIAS_LAYER : Layer number
        BIAS_DEVICE : Device (default: auto)
        
        Additional kwargs override environment values.
        """
        env_config = {
            "api_key": os.environ.get("NEURONPEDIA_API_KEY"),
            "model": os.environ.get("BIAS_MODEL", "gpt2"),
            "device": os.environ.get("BIAS_DEVICE", "auto"),
        }
        
        layer_env = os.environ.get("BIAS_LAYER")
        if layer_env:
            env_config["layer"] = int(layer_env)
        
        # Override with kwargs
        env_config.update(kwargs)
        
        return cls(**env_config)
    
    def __repr__(self) -> str:
        api_status = "configured" if self.api_key else "not set"
        return (
            f"BiasConfig(model='{self.model}', layer={self.layer}, "
            f"device='{self.device}', api_key={api_status})"
        )


def configure(
    api_key: Optional[str] = None,
    model: str = "gpt2",
    layer: Optional[int] = None,
    device: str = "auto",
    **kwargs
) -> BiasConfig:
    """
    Create a BiasConfig with common settings.
    
    This is a convenience function for creating configurations.
    
    Parameters
    ----------
    api_key : str, optional
        Your Neuronpedia API key
    model : str, default="gpt2"
        Model to use
    layer : int, optional
        Layer for steering
    device : str, default="auto"
        Device to use
    **kwargs
        Additional configuration options
    
    Returns
    -------
    BiasConfig
        Configuration object
    
    Examples
    --------
    >>> from bias import configure, Bias
    >>> 
    >>> config = configure(api_key="your-key", model="gpt2-medium")
    >>> bias = Bias(config=config)
    """
    return BiasConfig(
        api_key=api_key,
        model=model,
        layer=layer,
        device=device,
        **kwargs
    )


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
    base_url: str = "https://www.neuronpedia.org"
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

