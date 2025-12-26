"""
High-Level API for Bias
========================

This module provides the simplest interface for using Bias.
Just import and use with minimal configuration.

Examples
--------
>>> from bias import Bias, BiasConfig
>>> 
>>> # Method 1: Quick start - one line setup
>>> bias = Bias("gpt2")
>>> 
>>> # Method 2: With configuration object (recommended for API key)
>>> config = BiasConfig(
...     api_key="your-neuronpedia-api-key",
...     model="gpt2",
... )
>>> bias = Bias(config=config)
>>> 
>>> # Steer and generate
>>> bias.steer("professional writing")
>>> print(bias.generate("Write an email:"))

>>> # Even simpler - functional API
>>> from bias import steer, generate
>>> steer("formal tone")
>>> print(generate("Dear Sir,"))
"""

from typing import Optional, List, Dict, Any, Union
import torch

from bias.core.config import (
    BiasConfig,
    NeuronpediaConfig, 
    ModelConfig, 
    SteeringConfig,
    SUPPORTED_MODELS,
    get_model_config,
    configure,
)
from bias.core.engine import SteeringEngine
from bias.core.client import NeuronpediaClient
from bias.core.library import ConceptLibrary


# Global instance for functional API
_global_instance: Optional["Bias"] = None


class Bias:
    """
    High-level interface for LLM steering with Neuronpedia SAE features.
    
    This is the recommended entry point for most users. It provides
    a simple, intuitive API for steering language models.
    
    Parameters
    ----------
    model : str, optional
        Model name. Can be:
        - A shorthand like "gpt2", "gpt2-medium"
        - A HuggingFace model ID like "meta-llama/Llama-2-7b"
    config : BiasConfig, optional
        Configuration object with all settings. If provided,
        other parameters are ignored.
    layer : int, optional
        Which layer to apply steering to. If not specified,
        uses the recommended layer for the model.
    api_key : str, optional
        Neuronpedia API key for higher rate limits
    device : str, default="auto"
        Device to load model on ("auto", "cuda", "cpu", "mps")
    **kwargs
        Additional arguments passed to the engine
    
    Attributes
    ----------
    engine : SteeringEngine
        The underlying steering engine
    library : ConceptLibrary
        Library for saving/loading concepts
    config : BiasConfig
        The configuration being used
    
    Examples
    --------
    Basic usage:
    
    >>> bias = Bias("gpt2")
    >>> bias.steer("formal professional writing")
    >>> output = bias.generate("Write a business proposal:")
    >>> print(output)
    
    With configuration object (recommended):
    
    >>> from bias import BiasConfig, Bias
    >>> 
    >>> config = BiasConfig(
    ...     api_key="your-neuronpedia-api-key",
    ...     model="gpt2-medium",
    ...     layer=12,
    ...     device="cuda",
    ... )
    >>> bias = Bias(config=config)
    
    Using environment variable for API key:
    
    >>> # Set NEURONPEDIA_API_KEY environment variable
    >>> config = BiasConfig.from_env(model="gpt2")
    >>> bias = Bias(config=config)
    
    Using context manager (auto-cleanup):
    
    >>> with Bias("gpt2") as bias:
    ...     bias.steer("creative writing")
    ...     print(bias.generate("Once upon a time"))
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        config: Optional[BiasConfig] = None,
        layer: Optional[int] = None,
        api_key: Optional[str] = None,
        device: str = "auto",
        **kwargs
    ):
        # If config object is provided, use it directly
        if config is not None:
            self.config = config
        else:
            # Create config from individual parameters
            self.config = BiasConfig(
                api_key=api_key,
                model=model or "gpt2",
                layer=layer,
                device=device,
                **{k: v for k, v in kwargs.items() if k in BiasConfig.__dataclass_fields__}
            )
        
        # Get internal configs from BiasConfig
        self._neuronpedia_config = self.config.to_neuronpedia_config()
        self._model_config = self.config.to_model_config()
        
        # Initialize engine
        self.engine = SteeringEngine(
            model_name=self.config.hf_model_name,
            neuronpedia_config=self._neuronpedia_config,
            model_config=self._model_config,
        )
        
        # Initialize concept library
        self.library = ConceptLibrary(self.config.library_path)
        
        # Set as global instance
        global _global_instance
        _global_instance = self
    
    def steer(
        self,
        concept: str,
        intensity: float = 1.0,
        num_features: int = 5,
        save: bool = False,
    ) -> "Bias":
        """
        Steer the model toward a concept.
        
        Parameters
        ----------
        concept : str
            Natural language description of desired behavior
            (e.g., "formal writing", "humorous", "technical")
        intensity : float, default=1.0
            Steering strength. Higher values = stronger effect.
            Typical range: 0.5 to 5.0
        num_features : int, default=5
            Number of features to combine
        save : bool, default=False
            Whether to save this concept to the library
        
        Returns
        -------
        Bias
            Self, for method chaining
        
        Examples
        --------
        >>> bias.steer("professional").generate("Hello")
        >>> bias.steer("creative", intensity=2.0)
        """
        # Check library first
        cached = self.library.get_concept(
            concept,
            self._neuronpedia_config.model_id,
            self._neuronpedia_config.layer,
        )
        
        if cached:
            print(f"📚 Using cached features for '{concept}'")
            self.engine.steer_with_features(cached, intensities=intensity)
        else:
            self.engine.steer_with_concept(
                concept,
                intensity=intensity,
                num_features=num_features,
            )
            
            # Optionally save to library
            if save:
                features = self.engine.neuronpedia.search_features(
                    concept, top_k=num_features
                )
                if features:
                    self.library.add_concept(
                        concept=concept,
                        feature_ids=[f['id'] for f in features],
                        model_id=self._neuronpedia_config.model_id,
                        layer=self._neuronpedia_config.layer,
                    )
        
        return self
    
    def steer_features(
        self,
        feature_ids: List[int],
        intensity: float = 1.0,
    ) -> "Bias":
        """
        Steer using specific feature IDs.
        
        Parameters
        ----------
        feature_ids : List[int]
            List of Neuronpedia feature IDs
        intensity : float, default=1.0
            Steering strength
        
        Returns
        -------
        Bias
            Self, for method chaining
        """
        self.engine.steer_with_features(feature_ids, intensities=intensity)
        return self
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text with current steering.
        
        Parameters
        ----------
        prompt : str
            Input text to continue
        max_tokens : int, default=100
            Maximum tokens to generate
        temperature : float, default=0.7
            Sampling temperature
        **kwargs
            Additional generation arguments
        
        Returns
        -------
        str
            Generated text (including prompt)
        
        Examples
        --------
        >>> output = bias.generate("Write a poem about:", max_tokens=200)
        """
        return self.engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """
        Generate and return only the completion (without prompt).
        
        Parameters
        ----------
        prompt : str
            Input text
        max_tokens : int, default=100
            Maximum tokens to generate
        
        Returns
        -------
        str
            Only the generated completion
        """
        return self.engine.generate_only(prompt, max_tokens, **kwargs)
    
    def compare(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> Dict[str, str]:
        """
        Compare steered vs unsteered output.
        
        Returns
        -------
        Dict[str, str]
            Dictionary with 'steered' and 'unsteered' outputs
        """
        return self.engine.compare_outputs(prompt, max_tokens)
    
    def discover(
        self,
        concept: str,
        num_features: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Discover features for a concept without applying steering.
        
        Parameters
        ----------
        concept : str
            Concept to search for
        num_features : int, default=5
            Number of features to return
        
        Returns
        -------
        List[Dict]
            List of discovered features
        """
        return self.engine.neuronpedia.discover_features(
            concept,
            num_features=num_features,
            verbose=True,
        )
    
    def explore(
        self,
        feature_id: int,
        test_prompt: str = "Write a message: ",
    ) -> Dict[float, str]:
        """
        Explore a feature at different intensities.
        
        Parameters
        ----------
        feature_id : int
            Feature ID to explore
        test_prompt : str
            Prompt to test with
        
        Returns
        -------
        Dict[float, str]
            Outputs at different intensities
        """
        return self.engine.explore_feature(feature_id, test_prompt)
    
    def reset(self) -> "Bias":
        """
        Clear all steering and reset to baseline.
        
        Returns
        -------
        Bias
            Self, for method chaining
        """
        self.engine.clear_steering()
        return self
    
    def save_concept(
        self,
        name: str,
        feature_ids: List[int],
        notes: str = "",
    ):
        """
        Save a concept mapping to the library.
        
        Parameters
        ----------
        name : str
            Name for this concept
        feature_ids : List[int]
            Feature IDs to associate
        notes : str, optional
            Additional notes
        """
        self.library.add_concept(
            concept=name,
            feature_ids=feature_ids,
            model_id=self._neuronpedia_config.model_id,
            layer=self._neuronpedia_config.layer,
            notes=notes,
        )
    
    def load_concept(self, name: str) -> Optional[List[int]]:
        """
        Load feature IDs for a saved concept.
        
        Parameters
        ----------
        name : str
            Concept name
        
        Returns
        -------
        List[int] or None
            Feature IDs if found
        """
        return self.library.get_concept(
            name,
            self._neuronpedia_config.model_id,
            self._neuronpedia_config.layer,
        )
    
    def list_saved_concepts(self) -> List[str]:
        """List all saved concepts for the current model."""
        concepts = self.library.list_concepts(
            model_id=self._neuronpedia_config.model_id,
        )
        return [c['concept'] for c in concepts]
    
    @property
    def is_steering(self) -> bool:
        """Check if steering is currently active."""
        return self.engine.is_steering_active()
    
    def __enter__(self) -> "Bias":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clears steering."""
        self.reset()
        return False
    
    def __repr__(self) -> str:
        model = self._model_config.model_name
        layer = self._neuronpedia_config.layer
        active = "active" if self.is_steering else "inactive"
        return f"Bias(model='{model}', layer={layer}, steering={active})"


# =============================================================================
# Functional API
# =============================================================================

def _get_instance() -> Bias:
    """Get or create global Bias instance."""
    global _global_instance
    if _global_instance is None:
        _global_instance = Bias()
    return _global_instance


def steer(
    concept: str,
    intensity: float = 1.0,
    model: Optional[str] = None,
    **kwargs
) -> None:
    """
    Steer the model toward a concept (functional API).
    
    Parameters
    ----------
    concept : str
        Concept to steer toward
    intensity : float, default=1.0
        Steering strength
    model : str, optional
        Model to use (initializes new instance if different)
    
    Examples
    --------
    >>> from bias import steer, generate
    >>> steer("professional formal writing")
    >>> print(generate("Write a memo:"))
    """
    global _global_instance
    
    if model is not None and (
        _global_instance is None or 
        _global_instance._model_config.model_name != model
    ):
        _global_instance = Bias(model, **kwargs)
    
    _get_instance().steer(concept, intensity)


def generate(
    prompt: str,
    max_tokens: int = 100,
    **kwargs
) -> str:
    """
    Generate text with current steering (functional API).
    
    Parameters
    ----------
    prompt : str
        Input prompt
    max_tokens : int, default=100
        Maximum tokens to generate
    
    Returns
    -------
    str
        Generated text
    
    Examples
    --------
    >>> from bias import generate
    >>> output = generate("Hello, how are you?")
    """
    return _get_instance().generate(prompt, max_tokens, **kwargs)


def discover_features(
    concept: str,
    num_features: int = 5,
) -> List[Dict[str, Any]]:
    """
    Discover features for a concept (functional API).
    
    Parameters
    ----------
    concept : str
        Concept to search for
    num_features : int, default=5
        Number of features to return
    
    Returns
    -------
    List[Dict]
        List of discovered features
    """
    return _get_instance().discover(concept, num_features)


def reset_steering() -> None:
    """Clear all steering (functional API)."""
    if _global_instance is not None:
        _global_instance.reset()

