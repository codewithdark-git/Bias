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

### Method Chaining

```python
output = (
    Bias("gpt2")
    .steer("creative poetic", intensity=2.0)
    .generate("The moonlight danced upon")
)
```

### Compare Outputs

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
| `steer(concept, intensity)` | Steer toward a concept |
| `generate(prompt)` | Generate text |
| `compare(prompt)` | Compare steered vs unsteered |
| `discover(concept)` | Find features for a concept |
| `reset()` | Clear steering |

---

## Supported Models

| Model | Layer |
|-------|-------|
| `gpt2` | 6 |
| `gpt2-medium` | 12 |
| `gpt2-large` | 18 |
| `gpt2-xl` | 24 |

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
