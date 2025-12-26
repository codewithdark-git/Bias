"""
Concept Library
===============

Store and manage concept-to-feature mappings for reuse.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime


class ConceptLibrary:
    """
    Library for storing and retrieving concept-feature mappings.
    
    This class provides persistent storage for successful concept-feature
    mappings, allowing quick reuse without re-querying Neuronpedia.
    
    Attributes
    ----------
    cache_file : Path
        Path to the JSON file storing the library
    library : Dict
        In-memory dictionary of concept mappings
    
    Examples
    --------
    >>> from bias.core import ConceptLibrary
    >>> 
    >>> library = ConceptLibrary("my_concepts.json")
    >>> 
    >>> # Save a concept mapping
    >>> library.add_concept(
    ...     concept="professional",
    ...     feature_ids=[1234, 5678],
    ...     model_id="gpt2-small",
    ...     layer=6
    ... )
    >>> 
    >>> # Retrieve it later
    >>> features = library.get_concept("professional", "gpt2-small", 6)
    """
    
    def __init__(self, cache_file: str = "concept_library.json"):
        """
        Initialize the concept library.
        
        Parameters
        ----------
        cache_file : str
            Path to the JSON file for storing concepts
        """
        self.cache_file = Path(cache_file)
        self.library = self._load_library()
    
    def _load_library(self) -> Dict[str, Any]:
        """Load library from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load library: {e}")
                return {}
        return {}
    
    def save(self):
        """Save library to disk."""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.library, f, indent=2, ensure_ascii=False)
    
    def _make_key(
        self, 
        concept: str, 
        model_id: str, 
        layer: int
    ) -> str:
        """Create a unique key for a concept mapping."""
        return f"{model_id}::layer{layer}::{concept.lower().strip()}"
    
    def add_concept(
        self,
        concept: str,
        feature_ids: List[int],
        model_id: str,
        layer: int,
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add a concept mapping to the library.
        
        Parameters
        ----------
        concept : str
            The concept name/description
        feature_ids : List[int]
            List of Neuronpedia feature IDs
        model_id : str
            Neuronpedia model identifier
        layer : int
            The layer these features are from
        notes : str, optional
            Additional notes about this mapping
        metadata : Dict, optional
            Additional metadata to store
        """
        key = self._make_key(concept, model_id, layer)
        
        self.library[key] = {
            "concept": concept,
            "feature_ids": feature_ids,
            "model_id": model_id,
            "layer": layer,
            "notes": notes,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        
        self.save()
        print(f"✅ Saved concept '{concept}' with {len(feature_ids)} features")
    
    def get_concept(
        self,
        concept: str,
        model_id: str,
        layer: int,
    ) -> Optional[List[int]]:
        """
        Retrieve feature IDs for a concept.
        
        Parameters
        ----------
        concept : str
            The concept to look up
        model_id : str
            Neuronpedia model identifier
        layer : int
            The layer to look up
        
        Returns
        -------
        List[int] or None
            Feature IDs if found, None otherwise
        """
        key = self._make_key(concept, model_id, layer)
        
        if key in self.library:
            return self.library[key]["feature_ids"]
        return None
    
    def get_concept_full(
        self,
        concept: str,
        model_id: str,
        layer: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Get full concept entry including metadata.
        
        Returns
        -------
        Dict or None
            Full concept entry if found
        """
        key = self._make_key(concept, model_id, layer)
        return self.library.get(key)
    
    def update_concept(
        self,
        concept: str,
        model_id: str,
        layer: int,
        feature_ids: Optional[List[int]] = None,
        notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Update an existing concept mapping.
        
        Parameters
        ----------
        concept : str
            The concept to update
        model_id : str
            Neuronpedia model identifier
        layer : int
            The layer
        feature_ids : List[int], optional
            New feature IDs (if updating)
        notes : str, optional
            New notes (if updating)
        metadata : Dict, optional
            New metadata (if updating)
        """
        key = self._make_key(concept, model_id, layer)
        
        if key not in self.library:
            raise KeyError(f"Concept '{concept}' not found")
        
        if feature_ids is not None:
            self.library[key]["feature_ids"] = feature_ids
        if notes is not None:
            self.library[key]["notes"] = notes
        if metadata is not None:
            self.library[key]["metadata"].update(metadata)
        
        self.library[key]["updated_at"] = datetime.now().isoformat()
        self.save()
    
    def remove_concept(
        self,
        concept: str,
        model_id: str,
        layer: int,
    ) -> bool:
        """
        Remove a concept from the library.
        
        Returns
        -------
        bool
            True if concept was removed, False if not found
        """
        key = self._make_key(concept, model_id, layer)
        
        if key in self.library:
            del self.library[key]
            self.save()
            return True
        return False
    
    def list_concepts(
        self,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all concepts in the library.
        
        Parameters
        ----------
        model_id : str, optional
            Filter by model ID
        layer : int, optional
            Filter by layer
        
        Returns
        -------
        List[Dict]
            List of concept entries
        """
        results = []
        
        for entry in self.library.values():
            if model_id and entry['model_id'] != model_id:
                continue
            if layer is not None and entry['layer'] != layer:
                continue
            results.append(entry)
        
        return results
    
    def search_concepts(self, query: str) -> List[Dict[str, Any]]:
        """
        Search concepts by name.
        
        Parameters
        ----------
        query : str
            Search query (case-insensitive substring match)
        
        Returns
        -------
        List[Dict]
            Matching concept entries
        """
        query_lower = query.lower()
        return [
            entry for entry in self.library.values()
            if query_lower in entry['concept'].lower()
        ]
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the entire library as a dictionary."""
        return self.library.copy()
    
    def import_from_dict(
        self, 
        data: Dict[str, Any], 
        overwrite: bool = False
    ):
        """
        Import concepts from a dictionary.
        
        Parameters
        ----------
        data : Dict
            Dictionary of concept mappings
        overwrite : bool
            Whether to overwrite existing entries
        """
        for key, entry in data.items():
            if key in self.library and not overwrite:
                continue
            self.library[key] = entry
        
        self.save()
    
    def __len__(self) -> int:
        """Return number of concepts in library."""
        return len(self.library)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the library."""
        return key in self.library

