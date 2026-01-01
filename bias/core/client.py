"""
Neuronpedia API Client
======================

Client for interacting with Neuronpedia's SAE feature database.
Neuronpedia provides pre-trained Sparse Autoencoders with interpretable
features for various language models.
"""

import torch
import torch.nn.functional as F
import requests
from typing import List, Dict, Optional, Any
from bias.core.config import NeuronpediaConfig


class NeuronpediaClient:
    """
    Client for the Neuronpedia API.
    
    Neuronpedia hosts pre-trained Sparse Autoencoders (SAEs) with 
    interpretable features. Each feature corresponds to a specific
    concept or pattern that the model has learned.
    
    API Endpoints
    -------------
    - /feature/{model_id}/{layer}/{sae_id}/{feature_id} - Get feature details
    - /search - Search for features by description
    - /activations - Get feature activation patterns
    
    Attributes
    ----------
    config : NeuronpediaConfig
        Configuration for API access
    session : requests.Session
        HTTP session for API calls
    
    Examples
    --------
    >>> from bias.core import NeuronpediaClient, NeuronpediaConfig
    >>> 
    >>> config = NeuronpediaConfig(model_id="gpt2-small", layer=6)
    >>> client = NeuronpediaClient(config)
    >>> 
    >>> # Search for features
    >>> features = client.search_features("professional language")
    >>> print(features[0]['description'])
    
    >>> # Get steering vector
    >>> vector = client.get_feature_vector(features[0]['id'])
    """
    
    def __init__(self, config: NeuronpediaConfig):
        """
        Initialize the Neuronpedia client.
        
        Parameters
        ----------
        config : NeuronpediaConfig
            Configuration for API access
        """
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Bias-Python-Client/1.0.0",
            "Accept": "application/json",
        })
        
        if config.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {config.api_key}"
            })
    
    def search_features(
        self, 
        query: str, 
        top_k: int = 10,
        layer: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for features matching a concept description.
        
        Parameters
        ----------
        query : str
            Natural language description of the concept
            (e.g., "professional language", "positive sentiment")
        top_k : int, default=10
            Number of top matching features to return
        layer : int, optional
            Override the default layer from config
        
        Returns
        -------
        List[Dict]
            List of feature dictionaries containing:
            - id: Feature ID
            - description: Human-readable description
            - score: Relevance score
            - activating_examples: Example texts that activate this feature
        
        Examples
        --------
        >>> features = client.search_features("formal writing", top_k=5)
        >>> for f in features:
        ...     print(f"{f['id']}: {f['description']}")
        """
        # Neuronpedia API search endpoint - search by model
        url = f"{self.config.base_url}/api/explanation/search-model"
        
        # Build the request payload for POST request
        # API requires: modelId (string), query (string), offset (number, optional)
        payload = {
            "modelId": self.config.model_id,
            "query": query,
        }
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        
        try:
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            
            # Check for non-2xx status codes
            if response.status_code != 200:
                print(f"Neuronpedia API returned status {response.status_code}")
                # Try to get error details
                try:
                    error_detail = response.json()
                    print(f"Error details: {error_detail}")
                except (ValueError, requests.exceptions.JSONDecodeError):
                    print(f"Response: {response.text[:200] if response.text else 'Empty'}")
                return []
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                print(f"Unexpected content type: {content_type}")
                print(f"Response preview: {response.text[:200] if response.text else 'Empty'}")
                return []
            
            # Check for empty response
            if not response.content or not response.content.strip():
                print(f"Empty response from Neuronpedia API")
                return []
            
            data = response.json()
            
            # Handle different response formats
            # API returns: { results: [...], resultsCount, hasMore, nextOffset }
            if isinstance(data, dict):
                results = data.get('results', [])
                if isinstance(results, list):
                    # Limit to top_k results
                    features = results[:top_k]
                    # Normalize the response format
                    normalized = []
                    for f in features:
                        normalized.append({
                            'id': f.get('index', f.get('id')),
                            'description': f.get('description', ''),
                            'layer': f.get('layer', str(layer or self.config.layer)),
                            'modelId': f.get('modelId', self.config.model_id),
                            'score': f.get('score', 1.0),
                        })
                    return normalized
                return []
            elif isinstance(data, list):
                return data[:top_k]
            else:
                print(f"Unexpected response format: {type(data)}")
                return []
                
        except requests.exceptions.Timeout:
            print(f"Request to Neuronpedia timed out")
            return []
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error to Neuronpedia: {e}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error searching Neuronpedia: {e}")
            return []
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            print(f"Error parsing Neuronpedia response: {e}")
            if 'response' in locals() and response.text:
                print(f"Response preview: {response.text[:200]}")
            return []
    
    def get_feature_details(self, feature_id: int, layer: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific feature.
        
        Parameters
        ----------
        feature_id : int
            The feature ID (index) to look up
        layer : str, optional
            The layer/SAE ID (e.g., "6-res-jb"). If not provided, uses config.
        
        Returns
        -------
        Dict
            Feature details including:
            - id: Feature ID
            - description: Human-readable description
            - activating_examples: List of example texts
            - weight_vector: SAE decoder weights (steering vector)
            - activation_stats: Statistics about feature activations
        
        Examples
        --------
        >>> details = client.get_feature_details(1234)
        >>> print(details['description'])
        >>> print(f"Activates on: {details['activating_examples'][:3]}")
        """
        # Build the layer/SAE ID in the format expected by Neuronpedia
        # Format: {layer_num}-{sae_id}, e.g., "6-res-jb"
        if layer is None:
            layer = f"{self.config.layer}-{self.config.sae_id}"
        
        # Neuronpedia feature endpoint: GET /api/feature/{modelId}/{layer}/{index}
        url = (
            f"{self.config.base_url}/api/feature/"
            f"{self.config.model_id}/{layer}/{feature_id}"
        )
        
        # Set up headers
        headers = {}
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        
        try:
            response = self.session.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                print(f"Error fetching feature {feature_id}: HTTP {response.status_code}")
                return {}
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' not in content_type:
                print(f"Unexpected content type for feature {feature_id}: {content_type}")
                return {}
            
            if not response.content or not response.content.strip():
                print(f"Empty response for feature {feature_id}")
                return {}
            
            return response.json()
            
        except requests.exceptions.Timeout:
            print(f"Timeout fetching feature {feature_id}")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"Error fetching feature {feature_id}: {e}")
            return {}
        except (ValueError, requests.exceptions.JSONDecodeError) as e:
            print(f"Error parsing feature {feature_id} response: {e}")
            return {}
    
    def get_feature_vector(self, feature_id: int) -> Optional[torch.Tensor]:
        """
        Get the decoder weight vector for a feature.
        
        This vector is the steering direction in the model's hidden space.
        Adding this vector to activations will steer the model toward
        the concept represented by this feature.
        
        Parameters
        ----------
        feature_id : int
            The feature ID to get the vector for
        
        Returns
        -------
        torch.Tensor or None
            The steering vector, or None if not available
        
        Examples
        --------
        >>> vector = client.get_feature_vector(1234)
        >>> print(vector.shape)  # (hidden_dim,)
        """
        details = self.get_feature_details(feature_id)
        
        if 'weight_vector' in details:
            vector = torch.tensor(details['weight_vector'], dtype=torch.float32)
            return vector
        
        return None
    
    def get_multi_feature_vector(
        self, 
        feature_ids: List[int],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Combine multiple features into a single steering vector.
        
        This allows steering toward a combination of concepts,
        e.g., "professional" + "friendly" + "concise".
        
        Parameters
        ----------
        feature_ids : List[int]
            List of feature IDs to combine
        weights : List[float], optional
            Weights for each feature (default: equal weighting)
        normalize : bool, default=True
            Whether to normalize the combined vector
        
        Returns
        -------
        torch.Tensor or None
            Combined steering vector, or None if no vectors retrieved
        
        Examples
        --------
        >>> # Combine "professional" and "friendly" features
        >>> vector = client.get_multi_feature_vector(
        ...     feature_ids=[1234, 5678],
        ...     weights=[1.0, 0.5]
        ... )
        """
        if weights is None:
            weights = [1.0] * len(feature_ids)
        
        if len(weights) != len(feature_ids):
            raise ValueError("weights must have same length as feature_ids")
        
        vectors = []
        for fid, weight in zip(feature_ids, weights):
            vec = self.get_feature_vector(fid)
            if vec is not None:
                vectors.append(weight * vec)
        
        if not vectors:
            return None
        
        combined = sum(vectors)
        
        if normalize:
            combined = F.normalize(combined, dim=-1)
        
        return combined
    
    def discover_features(
        self, 
        concept: str,
        num_features: int = 5,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Discover and explain features related to a concept.
        
        This is a convenience method that searches for features and
        optionally prints detailed information about each one.
        
        Parameters
        ----------
        concept : str
            The concept to search for
        num_features : int, default=5
            Number of features to return
        verbose : bool, default=True
            Whether to print detailed information
        
        Returns
        -------
        List[Dict]
            List of feature dictionaries with scores
        
        Examples
        --------
        >>> features = client.discover_features("technical jargon")
        Found 5 relevant features:
        
        1. Feature #1234
           Description: Technical terminology and jargon
           Relevance Score: 0.892
        """
        if verbose:
            print(f"\n🔍 Searching for concept: '{concept}'")
        
        features = self.search_features(concept, top_k=num_features)
        
        if verbose and features:
            print(f"\n📊 Found {len(features)} relevant features:")
            for i, feature in enumerate(features, 1):
                print(f"\n  {i}. Feature #{feature.get('id', 'N/A')}")
                print(f"     Description: {feature.get('description', 'No description')}")
                print(f"     Relevance: {feature.get('score', 0):.3f}")
                
                if 'activating_examples' in feature:
                    examples = feature['activating_examples'][:2]
                    if examples:
                        print(f"     Examples:")
                        for ex in examples:
                            print(f"       • {ex[:60]}...")
        
        return features
    
    def validate_connection(self) -> bool:
        """
        Test the connection to Neuronpedia API.
        
        Returns
        -------
        bool
            True if connection is successful
        """
        try:
            # Test connection by checking the main page or API
            response = self.session.get(
                f"{self.config.base_url}",
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

