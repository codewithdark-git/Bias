"""
Helper Functions
================

Utility functions for the Bias library.
"""

from typing import Dict, List, Optional, Any


def format_output(text: str, max_width: int = 80) -> str:
    """
    Format text output with word wrapping.
    
    Parameters
    ----------
    text : str
        Text to format
    max_width : int
        Maximum line width
    
    Returns
    -------
    str
        Formatted text
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_width:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Parameters
    ----------
    text : str
        Text to truncate
    max_length : int
        Maximum length
    suffix : str
        Suffix to add when truncated
    
    Returns
    -------
    str
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def print_comparison(
    steered: str,
    unsteered: str,
    steered_label: str = "Steered",
    unsteered_label: str = "Unsteered",
) -> None:
    """
    Print a formatted comparison of steered vs unsteered outputs.
    
    Parameters
    ----------
    steered : str
        Steered output
    unsteered : str
        Unsteered output
    steered_label : str
        Label for steered output
    unsteered_label : str
        Label for unsteered output
    """
    separator = "=" * 60
    
    print(f"\n{separator}")
    print(f"{unsteered_label}:")
    print(separator)
    print(unsteered)
    
    print(f"\n{separator}")
    print(f"{steered_label}:")
    print(separator)
    print(steered)
    print()


def validate_model_name(model_name: str) -> bool:
    """
    Validate a model name.
    
    Parameters
    ----------
    model_name : str
        Model name to validate
    
    Returns
    -------
    bool
        True if valid
    """
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Basic validation
    if len(model_name) < 2:
        return False
    
    return True


def merge_feature_weights(
    feature_weights: List[Dict[str, Any]],
) -> Dict[int, float]:
    """
    Merge multiple feature weight specifications.
    
    Parameters
    ----------
    feature_weights : List[Dict]
        List of {feature_id: weight} dictionaries
    
    Returns
    -------
    Dict[int, float]
        Merged feature weights
    """
    merged = {}
    
    for fw in feature_weights:
        for fid, weight in fw.items():
            if fid in merged:
                merged[fid] += weight
            else:
                merged[fid] = weight
    
    return merged


def normalize_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """
    Normalize weights to sum to 1.0.
    
    Parameters
    ----------
    weights : Dict[int, float]
        Feature ID to weight mapping
    
    Returns
    -------
    Dict[int, float]
        Normalized weights
    """
    total = sum(weights.values())
    if total == 0:
        return weights
    return {k: v / total for k, v in weights.items()}

