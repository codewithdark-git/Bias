# Bias 🎯

> ***Steer Language Models with Interpretable SAE Features***

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://bias.readthedocs.io)

**Bias** is a Python library for steering LLM behavior using Sparse Autoencoder (SAE) features from [Neuronpedia](https://neuronpedia.org). Instead of prompt engineering or fine-tuning, simply describe the behavior you want.

---

## Installation

```bash
# From GitHub
pip install git+https://github.com/codewithdark-git/bias.git

# With dev tools
pip install "bias[dev] @ git+https://github.com/codewithdark-git/bias.git"
```

**Requirements:** Python 3.11+, PyTorch 2.5+

---

## Quick Start

### Basic Usage

```python
from bias import Bias

# Initialize
bias = Bias("gpt2")

# Steer toward a concept
bias.steer("professional formal writing", intensity=2.0)

# Generate
output = bias.generate("Write an email about the project:")
print(output)

# Reset
bias.reset()
```

### With Configuration (Recommended)

```python
from bias import Bias, BiasConfig

# Create configuration with your API key
config = BiasConfig(
    api_key="your-neuronpedia-api-key",  # Get from neuronpedia.org
    model="gpt2",
    device="auto",
)

# Pass config to Bias
bias = Bias(config=config)
bias.steer("professional writing")
output = bias.generate("Write an email:")
```

### Using Environment Variables

```bash
# Set your API key as environment variable
export NEURONPEDIA_API_KEY="your-api-key"
```

```python
from bias import Bias, BiasConfig

# Config automatically reads from environment
config = BiasConfig(model="gpt2")
bias = Bias(config=config)

# Or use from_env() for full environment configuration
config = BiasConfig.from_env()
bias = Bias(config=config)
```

---

## Configuration

### BiasConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | None | Neuronpedia API key (or use `NEURONPEDIA_API_KEY` env var) |
| `model` | str | "gpt2" | Model name ("gpt2", "gpt2-medium", etc.) |
| `layer` | int | auto | Steering layer (auto-selected per model) |
| `device` | str | "auto" | Device ("auto", "cuda", "cpu", "mps") |
| `dtype` | str | "float16" | Precision ("float16", "bfloat16", "float32") |
| `intensity` | float | 1.0 | Default steering intensity |
| `num_features` | int | 5 | Features per concept |

### Full Configuration Example

```python
from bias import Bias, BiasConfig

config = BiasConfig(
    # API Settings
    api_key="your-api-key",        # Neuronpedia API key
    
    # Model Settings
    model="gpt2-medium",           # Model to use
    layer=12,                      # Steering layer
    sae_id="res-jb",              # SAE variant
    
    # Device Settings
    device="cuda",                 # GPU acceleration
    dtype="float16",               # Half precision
    
    # Steering Defaults
    intensity=1.5,                 # Default intensity
    num_features=5,                # Features per concept
)

bias = Bias(config=config)
```

### Quick Configure Function

```python
from bias import configure, Bias

# One-liner configuration
config = configure(api_key="your-key", model="gpt2-medium")
bias = Bias(config=config)
```

---

## Method Chaining

```python
output = (
    Bias("gpt2")
    .steer("creative poetic", intensity=2.0)
    .generate("The moonlight danced upon")
)
```

## Compare Outputs

```python
bias.steer("formal academic", intensity=3.0)
results = bias.compare("Explain gravity:")
print("Unsteered:", results['unsteered'])
print("Steered:", results['steered'])
```

---

## CLI

```bash
# Generate with steering
bias generate "Write a poem:" -c "romantic" -i 2.0

# Discover features
bias discover "technical language"

# Interactive mode
bias interactive
```

---

## Core API

| Method | Description |
|--------|-------------|
| `Bias(config=config)` | Initialize with configuration object |
| `Bias(model, api_key=...)` | Initialize with parameters |
| `steer(concept, intensity)` | Steer toward a concept |
| `generate(prompt)` | Generate text |
| `compare(prompt)` | Compare steered vs unsteered |
| `discover(concept)` | Find features for a concept |
| `reset()` | Clear steering |

---

## Supported Models

| Model | Recommended Layer | Neuronpedia ID |
|-------|-------------------|----------------|
| `gpt2` | 6 | gpt2-small |
| `gpt2-medium` | 12 | gpt2-medium |
| `gpt2-large` | 18 | gpt2-large |
| `gpt2-xl` | 24 | gpt2-xl |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NEURONPEDIA_API_KEY` | Your Neuronpedia API key |
| `BIAS_MODEL` | Default model name |
| `BIAS_LAYER` | Default layer number |
| `BIAS_DEVICE` | Default device |

---

## How It Works

Bias uses **Sparse Autoencoder (SAE) features** from Neuronpedia to steer models. Each feature represents an interpretable concept (formality, sentiment, etc.). Adding these feature vectors to model activations shifts behavior toward that concept.

📖 **[Full Documentation](https://bias.readthedocs.io)** — Detailed guides on steering, SAEs, and the Neuronpedia integration.

---

## License

MIT License — see [LICENSE](LICENSE)

---

<p align="center">
  <strong>Made with 🎯 by <a href="https://github.com/codewithdark-git">codewithdark-git</a></strong>
</p>
