"""
AutoSteer with Neuronpedia API Integration

This system uses Neuronpedia's pre-computed SAE features for precise,
interpretable steering of language models.

Neuronpedia provides:
- Pre-trained Sparse Autoencoders for various models
- Interpretable features with human-readable descriptions
- Feature activations and examples
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple, Union
import requests
import json
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class NeuronpediaConfig:
    """Configuration for Neuronpedia API access"""
    api_key: Optional[str] = None  # Optional API key for rate limits
    base_url: str = "https://neuronpedia.org/api"
    model_id: str = "gpt2-small"  # Model identifier in Neuronpedia
    layer: int = 6  # Which layer's SAE to use
    sae_id: str = "res-jb"  # SAE variant identifier


class NeuronpediaClient:
    """
    Client for interacting with Neuronpedia API
    
    Neuronpedia API endpoints:
    - /feature/{model_id}/{layer}/{sae_id}/{feature_id} - Get feature details
    - /search - Search for features by description
    - /activations - Get feature activation patterns
    """
    
    def __init__(self, config: NeuronpediaConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})
    
    def search_features(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for features matching a concept description
        
        Args:
            query: Natural language description (e.g., "professional language")
            top_k: Number of top features to return
        
        Returns:
            List of feature dictionaries with id, description, activation stats
        """
        url = f"{self.config.base_url}/search"
        params = {
            "model_id": self.config.model_id,
            "layer": self.config.layer,
            "sae_id": self.config.sae_id,
            "query": query,
            "top_k": top_k
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error searching Neuronpedia: {e}")
            return []
    
    def get_feature_details(self, feature_id: int) -> Dict:
        """
        Get detailed information about a specific feature
        
        Returns:
            {
                'id': int,
                'description': str,
                'activating_examples': List[str],
                'weight_vector': List[float],  # SAE decoder weights
                'activation_stats': Dict
            }
        """
        url = f"{self.config.base_url}/feature/{self.config.model_id}/{self.config.layer}/{self.config.sae_id}/{feature_id}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching feature {feature_id}: {e}")
            return {}
    
    def get_feature_vector(self, feature_id: int) -> Optional[torch.Tensor]:
        """
        Get the decoder weight vector for a feature
        
        This is the steering vector in the model's hidden space
        """
        details = self.get_feature_details(feature_id)
        
        if 'weight_vector' in details:
            vector = torch.tensor(details['weight_vector'], dtype=torch.float32)
            return vector
        
        return None
    
    def get_multi_feature_vector(
        self, 
        feature_ids: List[int],
        weights: Optional[List[float]] = None
    ) -> Optional[torch.Tensor]:
        """
        Combine multiple features into a single steering vector
        
        Args:
            feature_ids: List of feature IDs to combine
            weights: Optional weights for each feature (default: equal weighting)
        
        Returns:
            Combined steering vector
        """
        if weights is None:
            weights = [1.0] * len(feature_ids)
        
        vectors = []
        for fid, weight in zip(feature_ids, weights):
            vec = self.get_feature_vector(fid)
            if vec is not None:
                vectors.append(weight * vec)
        
        if not vectors:
            return None
        
        combined = sum(vectors)
        return F.normalize(combined, dim=-1)
    
    def discover_concept_features(
        self, 
        concept: str,
        num_features: int = 5,
        explain: bool = True
    ) -> List[Dict]:
        """
        Discover and explain features related to a concept
        
        Args:
            concept: Concept to search for
            num_features: Number of features to return
            explain: Whether to print explanations
        
        Returns:
            List of feature dictionaries with scores
        """
        print(f"\nSearching Neuronpedia for concept: '{concept}'")
        features = self.search_features(concept, top_k=num_features)
        
        if explain and features:
            print(f"\nFound {len(features)} relevant features:")
            for i, feature in enumerate(features, 1):
                print(f"\n{i}. Feature #{feature.get('id', 'N/A')}")
                print(f"   Description: {feature.get('description', 'No description')}")
                print(f"   Relevance Score: {feature.get('score', 0):.3f}")
                
                if 'activating_examples' in feature:
                    examples = feature['activating_examples'][:3]
                    print(f"   Example activations:")
                    for ex in examples:
                        print(f"     - {ex}")
        
        return features


class NeuronpediaSteeringEngine:
    """
    Steering engine that uses Neuronpedia SAE features
    
    This provides interpretable, feature-level control over model behavior
    """
    
    def __init__(
        self, 
        model_name: str,
        neuronpedia_config: NeuronpediaConfig,
        device: str = "auto"
    ):
        print(f"Initializing Neuronpedia Steering Engine")
        print(f"Model: {model_name}")
        print(f"Neuronpedia Model ID: {neuronpedia_config.model_id}")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float16
        )
        self.device = next(self.model.parameters()).device
        
        # Initialize Neuronpedia client
        self.neuronpedia = NeuronpediaClient(neuronpedia_config)
        self.config = neuronpedia_config
        
        # Steering state
        self.active_features = {}  # layer -> {feature_id: intensity}
        self.hook_handles = []
        
        print(f"Model loaded on {self.device}")
        print(f"Target layer for steering: {neuronpedia_config.layer}")
    
    def steer_with_concept(
        self,
        concept: str,
        intensity: float = 1.0,
        num_features: int = 5,
        auto_balance: bool = True
    ):
        """
        Automatically find and apply features for a concept
        
        Args:
            concept: Natural language concept (e.g., "formal business language")
            intensity: Overall steering strength
            num_features: How many features to use
            auto_balance: Whether to balance feature weights by relevance
        """
        print(f"\n{'='*80}")
        print(f"Setting up steering for concept: '{concept}'")
        print(f"{'='*80}")
        
        # Discover relevant features
        features = self.neuronpedia.discover_concept_features(
            concept, 
            num_features=num_features,
            explain=True
        )
        
        if not features:
            print(f"⚠️  No features found for '{concept}'")
            return False
        
        # Extract feature IDs and scores
        feature_ids = [f['id'] for f in features]
        
        if auto_balance:
            # Weight by relevance scores
            scores = [f.get('score', 1.0) for f in features]
            total_score = sum(scores)
            weights = [s / total_score for s in scores] if total_score > 0 else [1.0/len(scores)] * len(scores)
        else:
            weights = [1.0] * len(feature_ids)
        
        # Get combined steering vector
        print(f"\nCombining {len(feature_ids)} features...")
        steering_vector = self.neuronpedia.get_multi_feature_vector(feature_ids, weights)
        
        if steering_vector is None:
            print("⚠️  Failed to retrieve feature vectors")
            return False
        
        # Apply steering
        self.apply_feature_steering(
            layer=self.config.layer,
            steering_vector=steering_vector,
            intensity=intensity
        )
        
        print(f"✓ Steering applied successfully")
        return True
    
    def steer_with_features(
        self,
        feature_ids: List[int],
        intensities: Union[float, List[float]] = 1.0,
        layer: Optional[int] = None
    ):
        """
        Directly steer using specific feature IDs
        
        Args:
            feature_ids: List of Neuronpedia feature IDs
            intensities: Single intensity or per-feature intensities
            layer: Target layer (default: from config)
        """
        if layer is None:
            layer = self.config.layer
        
        # Handle intensity input
        if isinstance(intensities, (int, float)):
            intensities = [float(intensities)] * len(feature_ids)
        
        print(f"\nApplying {len(feature_ids)} features to layer {layer}")
        
        # Get combined vector
        steering_vector = self.neuronpedia.get_multi_feature_vector(
            feature_ids, 
            intensities
        )
        
        if steering_vector is None:
            print("⚠️  Failed to retrieve feature vectors")
            return False
        
        self.apply_feature_steering(layer, steering_vector, intensity=1.0)
        return True
    
    def apply_feature_steering(
        self,
        layer: int,
        steering_vector: torch.Tensor,
        intensity: float = 1.0
    ):
        """
        Apply a steering vector to a specific layer
        """
        self.clear_steering()
        
        # Move vector to device
        steering_vector = steering_vector.to(self.device).to(self.model.dtype)
        
        # Create and register hook
        def make_hook(vector, intensity):
            def hook(module, input, output):
                # output[0] shape: (batch_size, seq_len, hidden_dim)
                output[0][:, -1, :] += intensity * vector
                return output
            return hook
        
        target_layer = self.model.model.layers[layer]
        handle = target_layer.register_forward_hook(
            make_hook(steering_vector, intensity)
        )
        self.hook_handles.append(handle)
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text with active steering"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                **kwargs
            )
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    def compare_outputs(
        self, 
        prompt: str,
        max_new_tokens: int = 100
    ) -> Dict[str, str]:
        """Compare steered vs unsteered output"""
        # Generate with steering
        steered = self.generate(prompt, max_new_tokens)
        
        # Temporarily disable
        temp_handles = self.hook_handles.copy()
        self.clear_steering()
        
        # Generate without steering
        unsteered = self.generate(prompt, max_new_tokens)
        
        # Restore
        self.hook_handles = temp_handles
        
        return {"unsteered": unsteered, "steered": steered}
    
    def clear_steering(self):
        """Remove all active steering"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.active_features = {}
    
    def explore_feature(self, feature_id: int):
        """
        Interactively explore a feature's behavior
        """
        print(f"\n{'='*80}")
        print(f"Exploring Feature #{feature_id}")
        print(f"{'='*80}")
        
        details = self.neuronpedia.get_feature_details(feature_id)
        
        if not details:
            print("⚠️  Feature not found")
            return
        
        print(f"\nDescription: {details.get('description', 'N/A')}")
        
        if 'activating_examples' in details:
            print(f"\nExamples where this feature activates:")
            for i, example in enumerate(details['activating_examples'][:5], 1):
                print(f"{i}. {example}")
        
        # Test at different intensities
        print(f"\nTesting feature at different intensities:")
        test_prompt = "Write a brief message: "
        
        for intensity in [0.0, 1.0, 3.0, 5.0]:
            if intensity == 0.0:
                self.clear_steering()
            else:
                self.steer_with_features([feature_id], intensities=intensity)
            
            output = self.generate(test_prompt, max_new_tokens=50)
            print(f"\nIntensity {intensity}:")
            print(f"  {output}")
        
        self.clear_steering()


class ConceptLibrary:
    """
    Pre-defined concept-to-feature mappings
    
    This class stores successful concept-feature mappings for quick reuse
    """
    
    def __init__(self, cache_file: str = "concept_library.json"):
        self.cache_file = cache_file
        self.library = self._load_library()
    
    def _load_library(self) -> Dict:
        """Load library from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_library(self):
        """Save library to disk"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.library, f, indent=2)
    
    def add_concept(
        self, 
        concept: str,
        feature_ids: List[int],
        model_id: str,
        layer: int,
        notes: str = ""
    ):
        """Add a concept mapping to the library"""
        key = f"{model_id}_layer{layer}_{concept}"
        self.library[key] = {
            "concept": concept,
            "feature_ids": feature_ids,
            "model_id": model_id,
            "layer": layer,
            "notes": notes
        }
        self.save_library()
    
    def get_concept(
        self, 
        concept: str, 
        model_id: str, 
        layer: int
    ) -> Optional[List[int]]:
        """Retrieve feature IDs for a concept"""
        key = f"{model_id}_layer{layer}_{concept}"
        if key in self.library:
            return self.library[key]["feature_ids"]
        return None
    
    def list_concepts(self, model_id: Optional[str] = None) -> List[str]:
        """List all available concepts"""
        if model_id:
            return [
                v['concept'] for k, v in self.library.items() 
                if v['model_id'] == model_id
            ]
        return [v['concept'] for v in self.library.values()]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def basic_neuronpedia_example():
    """Basic usage with Neuronpedia integration"""
    print("\n" + "="*80)
    print("Example 1: Basic Neuronpedia Steering")
    print("="*80)
    
    # Configure Neuronpedia access
    neuronpedia_config = NeuronpediaConfig(
        model_id="gpt2-small",
        layer=6,
        sae_id="res-jb"
    )
    
    # Initialize engine
    engine = NeuronpediaSteeringEngine(
        model_name="gpt2",
        neuronpedia_config=neuronpedia_config
    )
    
    # Steer with a concept
    engine.steer_with_concept(
        concept="formal professional writing",
        intensity=2.0,
        num_features=5
    )
    
    # Generate
    prompt = "Write an email about the quarterly results:"
    output = engine.generate(prompt, max_new_tokens=100)
    
    print(f"\n\nPrompt: {prompt}")
    print(f"Output: {output}")
    
    engine.clear_steering()


def multi_concept_neuronpedia_example():
    """Combine multiple concepts using different features"""
    print("\n" + "="*80)
    print("Example 2: Multi-Concept Steering with Neuronpedia")
    print("="*80)
    
    neuronpedia_config = NeuronpediaConfig(
        model_id="gpt2-small",
        layer=6,
        sae_id="res-jb"
    )
    
    engine = NeuronpediaSteeringEngine(
        model_name="gpt2",
        neuronpedia_config=neuronpedia_config
    )
    
    # Search for multiple concepts
    concepts = ["professional", "concise", "friendly"]
    
    all_features = []
    for concept in concepts:
        features = engine.neuronpedia.discover_concept_features(
            concept, 
            num_features=3,
            explain=True
        )
        all_features.extend([f['id'] for f in features[:2]])
    
    # Apply combined steering
    engine.steer_with_features(
        feature_ids=all_features,
        intensities=1.5
    )
    
    prompt = "Respond to a customer inquiry about shipping:"
    output = engine.generate(prompt, max_new_tokens=80)
    
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")
    
    engine.clear_steering()


def feature_exploration_example():
    """Explore individual features interactively"""
    print("\n" + "="*80)
    print("Example 3: Feature Exploration")
    print("="*80)
    
    neuronpedia_config = NeuronpediaConfig(
        model_id="gpt2-small",
        layer=6,
        sae_id="res-jb"
    )
    
    engine = NeuronpediaSteeringEngine(
        model_name="gpt2",
        neuronpedia_config=neuronpedia_config
    )
    
    # Find features for a concept
    features = engine.neuronpedia.discover_concept_features(
        "technical language",
        num_features=5,
        explain=True
    )
    
    if features:
        # Explore the top feature
        top_feature_id = features[0]['id']
        engine.explore_feature(top_feature_id)


def concept_library_example():
    """Build and use a concept library"""
    print("\n" + "="*80)
    print("Example 4: Using Concept Library")
    print("="*80)
    
    library = ConceptLibrary()
    
    neuronpedia_config = NeuronpediaConfig(
        model_id="gpt2-small",
        layer=6,
        sae_id="res-jb"
    )
    
    engine = NeuronpediaSteeringEngine(
        model_name="gpt2",
        neuronpedia_config=neuronpedia_config
    )
    
    concept = "professional"
    
    # Check if concept exists in library
    cached_features = library.get_concept(
        concept=concept,
        model_id=neuronpedia_config.model_id,
        layer=neuronpedia_config.layer
    )
    
    if cached_features:
        print(f"Using cached features for '{concept}'")
        engine.steer_with_features(cached_features, intensities=2.0)
    else:
        print(f"Discovering new features for '{concept}'")
        features = engine.neuronpedia.discover_concept_features(
            concept, 
            num_features=5
        )
        
        feature_ids = [f['id'] for f in features]
        
        # Save to library
        library.add_concept(
            concept=concept,
            feature_ids=feature_ids,
            model_id=neuronpedia_config.model_id,
            layer=neuronpedia_config.layer,
            notes="Optimized for business writing"
        )
        
        engine.steer_with_features(feature_ids, intensities=2.0)
    
    # Generate
    prompt = "Write a business proposal:"
    output = engine.generate(prompt, max_new_tokens=100)
    
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")


def interactive_neuronpedia_cli():
    """Interactive CLI with Neuronpedia"""
    print("\n" + "="*80)
    print("AutoSteer with Neuronpedia - Interactive Mode")
    print("="*80)
    
    # Configuration
    model_name = input("\nEnter model name (e.g., 'gpt2'): ").strip() or "gpt2"
    neuronpedia_model_id = input("Enter Neuronpedia model ID (e.g., 'gpt2-small'): ").strip() or "gpt2-small"
    layer = int(input("Enter target layer (e.g., 6): ").strip() or "6")
    
    neuronpedia_config = NeuronpediaConfig(
        model_id=neuronpedia_model_id,
        layer=layer,
        sae_id="res-jb"
    )
    
    engine = NeuronpediaSteeringEngine(
        model_name=model_name,
        neuronpedia_config=neuronpedia_config
    )
    
    print("\n" + "="*80)
    print("Commands:")
    print("  concept <text> - Steer with a concept")
    print("  features <id1,id2,...> - Steer with specific feature IDs")
    print("  explore <id> - Explore a feature")
    print("  compare - Compare steered vs unsteered")
    print("  clear - Clear steering")
    print("  quit - Exit")
    print("="*80)
    
    while True:
        command = input("\n> ").strip()
        
        if command.lower() == "quit":
            break
        
        elif command.lower().startswith("concept "):
            concept = command[8:].strip()
            intensity = float(input("Intensity (1.0-5.0): ").strip() or "2.0")
            engine.steer_with_concept(concept, intensity=intensity)
        
        elif command.lower().startswith("features "):
            ids_str = command[9:].strip()
            feature_ids = [int(x.strip()) for x in ids_str.split(",")]
            intensity = float(input("Intensity (1.0-5.0): ").strip() or "2.0")
            engine.steer_with_features(feature_ids, intensities=intensity)
        
        elif command.lower().startswith("explore "):
            feature_id = int(command[8:].strip())
            engine.explore_feature(feature_id)
        
        elif command.lower() == "compare":
            prompt = input("Enter prompt: ")
            results = engine.compare_outputs(prompt)
            print(f"\n{'='*40}\nUNSTEERED:\n{'='*40}")
            print(results['unsteered'])
            print(f"\n{'='*40}\nSTEERED:\n{'='*40}")
            print(results['steered'])
        
        elif command.lower() == "clear":
            engine.clear_steering()
            print("✓ Steering cleared")
        
        else:
            # Treat as prompt
            output = engine.generate(command, max_new_tokens=100)
            print(f"\n{output}")
    
    engine.clear_steering()
    print("\nGoodbye!")


if __name__ == "__main__":
    print("""
    AutoSteer with Neuronpedia Integration
    ======================================
    
    This system uses Neuronpedia's interpretable SAE features for steering.
    
    Note: Neuronpedia API endpoints used in this code are illustrative.
    Check https://neuronpedia.org/api-doc for actual API documentation.
    
    """)
    
    # Run interactive CLI
    interactive_neuronpedia_cli()
    
    # Or run examples
    # basic_neuronpedia_example()
    # multi_concept_neuronpedia_example()
    # feature_exploration_example()
    # concept_library_example()