"""
Steering Engine
===============

The core engine for applying SAE-based steering to language models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple, Union, Any

from bias.core.config import NeuronpediaConfig, ModelConfig, SteeringConfig
from bias.core.client import NeuronpediaClient


class SteeringEngine:
    """
    Main engine for steering language models using SAE features.
    
    The SteeringEngine loads a language model and uses Neuronpedia's
    SAE features to steer the model's behavior. It works by adding
    steering vectors to the model's hidden states during generation.
    
    How it Works
    ------------
    1. Features from Neuronpedia represent learned concepts (e.g., "formal")
    2. Each feature has a decoder weight vector in the model's hidden space
    3. Adding this vector to activations steers the model toward that concept
    4. Multiple features can be combined for nuanced steering
    
    Attributes
    ----------
    model : AutoModelForCausalLM
        The loaded language model
    tokenizer : AutoTokenizer
        The model's tokenizer
    neuronpedia : NeuronpediaClient
        Client for fetching SAE features
    device : torch.device
        Device the model is loaded on
    
    Examples
    --------
    >>> from bias.core import SteeringEngine, NeuronpediaConfig
    >>> 
    >>> config = NeuronpediaConfig(model_id="gpt2-small", layer=6)
    >>> engine = SteeringEngine("gpt2", neuronpedia_config=config)
    >>> 
    >>> # Steer with a concept
    >>> engine.steer_with_concept("professional formal writing", intensity=2.0)
    >>> 
    >>> # Generate text
    >>> output = engine.generate("Write an email:")
    >>> print(output)
    """
    
    def __init__(
        self,
        model_name: str,
        neuronpedia_config: NeuronpediaConfig,
        model_config: Optional[ModelConfig] = None,
        steering_config: Optional[SteeringConfig] = None,
    ):
        """
        Initialize the steering engine.
        
        Parameters
        ----------
        model_name : str
            HuggingFace model name or path (e.g., "gpt2", "meta-llama/Llama-2-7b")
        neuronpedia_config : NeuronpediaConfig
            Configuration for Neuronpedia API access
        model_config : ModelConfig, optional
            Configuration for model loading
        steering_config : SteeringConfig, optional
            Configuration for steering behavior
        """
        self.model_config = model_config or ModelConfig(model_name=model_name)
        self.steering_config = steering_config or SteeringConfig()
        self.neuronpedia_config = neuronpedia_config
        
        print(f"🚀 Initializing Bias Steering Engine")
        print(f"   Model: {model_name}")
        print(f"   Neuronpedia ID: {neuronpedia_config.model_id}")
        print(f"   Target Layer: {neuronpedia_config.layer}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.model_config.dtype, torch.float16)
        
        # Load model
        load_kwargs = {
            "device_map": self.model_config.device,
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
        }
        
        if self.model_config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self.model_config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        self.device = next(self.model.parameters()).device
        
        # Initialize Neuronpedia client
        self.neuronpedia = NeuronpediaClient(neuronpedia_config)
        
        # Steering state
        self._hook_handles: List = []
        self._active_steering: Dict[int, Dict] = {}  # layer -> steering info
        
        print(f"✅ Engine ready on {self.device}")
    
    def steer_with_concept(
        self,
        concept: str,
        intensity: float = 1.0,
        num_features: int = 5,
        auto_balance: bool = True,
        layer: Optional[int] = None,
    ) -> bool:
        """
        Automatically find and apply features for a concept.
        
        This is the easiest way to steer the model - just describe
        the behavior you want in natural language.
        
        Parameters
        ----------
        concept : str
            Natural language description of desired behavior
            (e.g., "formal business language", "friendly and casual")
        intensity : float, default=1.0
            Steering strength (higher = stronger effect)
        num_features : int, default=5
            How many features to combine
        auto_balance : bool, default=True
            Weight features by their relevance scores
        layer : int, optional
            Override the target layer
        
        Returns
        -------
        bool
            True if steering was applied successfully
        
        Examples
        --------
        >>> engine.steer_with_concept("professional writing", intensity=2.0)
        >>> engine.steer_with_concept("humorous and witty", intensity=1.5)
        """
        target_layer = layer or self.neuronpedia_config.layer
        
        print(f"\n{'='*60}")
        print(f"🎯 Setting up steering for: '{concept}'")
        print(f"{'='*60}")
        
        # Discover relevant features
        features = self.neuronpedia.discover_features(
            concept,
            num_features=num_features,
            verbose=True
        )
        
        if not features:
            print(f"⚠️  No features found for '{concept}'")
            return False
        
        # Extract IDs and compute weights
        feature_ids = [f['id'] for f in features]
        
        if auto_balance:
            scores = [f.get('score', 1.0) for f in features]
            total = sum(scores)
            weights = [s / total for s in scores] if total > 0 else None
        else:
            weights = None
        
        # Get combined steering vector
        print(f"\n📐 Combining {len(feature_ids)} features...")
        steering_vector = self.neuronpedia.get_multi_feature_vector(
            feature_ids,
            weights=weights
        )
        
        if steering_vector is None:
            print("⚠️  Failed to retrieve feature vectors")
            return False
        
        # Apply steering
        self._apply_steering_vector(
            layer=target_layer,
            steering_vector=steering_vector,
            intensity=intensity
        )
        
        print(f"✅ Steering applied (intensity={intensity})")
        return True
    
    def steer_with_features(
        self,
        feature_ids: List[int],
        intensities: Union[float, List[float]] = 1.0,
        layer: Optional[int] = None,
    ) -> bool:
        """
        Steer using specific feature IDs.
        
        Use this when you know the exact features you want to use,
        e.g., from previous exploration or saved configurations.
        
        Parameters
        ----------
        feature_ids : List[int]
            List of Neuronpedia feature IDs
        intensities : float or List[float], default=1.0
            Single intensity or per-feature intensities
        layer : int, optional
            Target layer (default: from config)
        
        Returns
        -------
        bool
            True if steering was applied successfully
        
        Examples
        --------
        >>> engine.steer_with_features([1234, 5678], intensities=2.0)
        >>> engine.steer_with_features([1234, 5678], intensities=[2.0, 1.5])
        """
        target_layer = layer or self.neuronpedia_config.layer
        
        # Handle intensity input
        if isinstance(intensities, (int, float)):
            weights = [float(intensities)] * len(feature_ids)
        else:
            weights = list(intensities)
        
        print(f"\n🎯 Applying {len(feature_ids)} features to layer {target_layer}")
        
        # Get combined vector
        steering_vector = self.neuronpedia.get_multi_feature_vector(
            feature_ids,
            weights=weights
        )
        
        if steering_vector is None:
            print("⚠️  Failed to retrieve feature vectors")
            return False
        
        self._apply_steering_vector(target_layer, steering_vector, intensity=1.0)
        return True
    
    def steer_with_vector(
        self,
        steering_vector: torch.Tensor,
        intensity: float = 1.0,
        layer: Optional[int] = None,
    ):
        """
        Apply a custom steering vector directly.
        
        Parameters
        ----------
        steering_vector : torch.Tensor
            The steering direction vector
        intensity : float, default=1.0
            Steering strength
        layer : int, optional
            Target layer
        """
        target_layer = layer or self.neuronpedia_config.layer
        self._apply_steering_vector(target_layer, steering_vector, intensity)
    
    def _apply_steering_vector(
        self,
        layer: int,
        steering_vector: torch.Tensor,
        intensity: float = 1.0,
    ):
        """Internal method to apply steering vector to a layer."""
        # Clear existing steering first
        self.clear_steering()
        
        # Move vector to correct device and dtype
        steering_vector = steering_vector.to(self.device).to(self.model.dtype)
        
        # Create hook function
        def make_hook(vector: torch.Tensor, strength: float):
            def hook(module, input, output):
                # output[0] shape: (batch_size, seq_len, hidden_dim)
                # Apply steering to the last token position
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    hidden_states[:, -1, :] += strength * vector
                    return (hidden_states,) + output[1:]
                else:
                    output[:, -1, :] += strength * vector
                    return output
            return hook
        
        # Get target layer - handle different model architectures
        target_module = self._get_layer_module(layer)
        
        # Register hook
        handle = target_module.register_forward_hook(
            make_hook(steering_vector, intensity)
        )
        self._hook_handles.append(handle)
        
        # Track active steering
        self._active_steering[layer] = {
            "vector": steering_vector,
            "intensity": intensity,
        }
    
    def _get_layer_module(self, layer: int):
        """Get the transformer layer module for different architectures."""
        # Try common architectures
        if hasattr(self.model, 'transformer'):
            # GPT-2 style
            return self.model.transformer.h[layer]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA style
            return self.model.model.layers[layer]
        elif hasattr(self.model, 'gpt_neox'):
            # GPT-NeoX style
            return self.model.gpt_neox.layers[layer]
        else:
            raise ValueError(
                f"Unknown model architecture. Cannot find layer {layer}. "
                f"Model type: {type(self.model)}"
            )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with active steering.
        
        Parameters
        ----------
        prompt : str
            The input prompt
        max_new_tokens : int, default=100
            Maximum tokens to generate
        temperature : float, default=0.7
            Sampling temperature (higher = more random)
        top_p : float, default=0.9
            Nucleus sampling probability
        do_sample : bool, default=True
            Whether to use sampling (False = greedy)
        **kwargs
            Additional arguments passed to model.generate()
        
        Returns
        -------
        str
            The generated text (including the prompt)
        
        Examples
        --------
        >>> output = engine.generate(
        ...     "Write a short story about:",
        ...     max_new_tokens=200,
        ...     temperature=0.8
        ... )
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def generate_only(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """
        Generate text and return only the new tokens (without prompt).
        
        Parameters
        ----------
        prompt : str
            The input prompt
        max_new_tokens : int, default=100
            Maximum tokens to generate
        **kwargs
            Additional arguments passed to generate()
        
        Returns
        -------
        str
            Only the newly generated text
        """
        full_output = self.generate(prompt, max_new_tokens, **kwargs)
        # Remove the prompt from the output
        if full_output.startswith(prompt):
            return full_output[len(prompt):].strip()
        return full_output
    
    def compare_outputs(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Dict[str, str]:
        """
        Compare steered vs unsteered outputs.
        
        This is useful for seeing the effect of steering.
        
        Parameters
        ----------
        prompt : str
            The input prompt
        max_new_tokens : int, default=100
            Maximum tokens to generate
        
        Returns
        -------
        Dict[str, str]
            Dictionary with 'unsteered' and 'steered' outputs
        
        Examples
        --------
        >>> results = engine.compare_outputs("Write a message:")
        >>> print("Without steering:", results['unsteered'])
        >>> print("With steering:", results['steered'])
        """
        # Generate with current steering
        steered = self.generate(prompt, max_new_tokens, **kwargs)
        
        # Temporarily save and clear steering
        saved_handles = self._hook_handles.copy()
        saved_steering = self._active_steering.copy()
        self.clear_steering()
        
        # Generate without steering
        unsteered = self.generate(prompt, max_new_tokens, **kwargs)
        
        # Restore steering
        self._hook_handles = saved_handles
        self._active_steering = saved_steering
        
        return {
            "unsteered": unsteered,
            "steered": steered
        }
    
    def clear_steering(self):
        """
        Remove all active steering.
        
        Call this to reset the model to its original behavior.
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._active_steering = {}
    
    def get_active_steering(self) -> Dict[int, Dict]:
        """
        Get information about currently active steering.
        
        Returns
        -------
        Dict[int, Dict]
            Dictionary mapping layer numbers to steering info
        """
        return self._active_steering.copy()
    
    def is_steering_active(self) -> bool:
        """Check if any steering is currently active."""
        return len(self._hook_handles) > 0
    
    def explore_feature(
        self,
        feature_id: int,
        test_prompt: str = "Write a brief message: ",
        intensities: List[float] = [0.0, 1.0, 3.0, 5.0],
    ) -> Dict[float, str]:
        """
        Explore a feature's behavior at different intensities.
        
        Parameters
        ----------
        feature_id : int
            The feature ID to explore
        test_prompt : str
            Prompt to use for testing
        intensities : List[float]
            Intensities to test
        
        Returns
        -------
        Dict[float, str]
            Dictionary mapping intensity to generated output
        """
        print(f"\n{'='*60}")
        print(f"🔬 Exploring Feature #{feature_id}")
        print(f"{'='*60}")
        
        # Get feature details
        details = self.neuronpedia.get_feature_details(feature_id)
        
        if not details:
            print("⚠️  Feature not found")
            return {}
        
        print(f"\n📝 Description: {details.get('description', 'N/A')}")
        
        if 'activating_examples' in details:
            print(f"\n💡 Example activations:")
            for ex in details['activating_examples'][:3]:
                print(f"   • {ex}")
        
        # Test at different intensities
        print(f"\n🧪 Testing at different intensities...")
        results = {}
        
        for intensity in intensities:
            if intensity == 0.0:
                self.clear_steering()
            else:
                self.steer_with_features([feature_id], intensities=intensity)
            
            output = self.generate_only(test_prompt, max_new_tokens=50)
            results[intensity] = output
            
            print(f"\n   Intensity {intensity}:")
            print(f"   {output[:100]}...")
        
        self.clear_steering()
        return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clears steering."""
        self.clear_steering()
        return False

