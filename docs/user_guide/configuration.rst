Configuration Guide
===================

This guide covers all configuration options in Bias.

Bias Class Configuration
------------------------

The ``Bias`` class accepts these parameters:

.. code-block:: python

   from bias import Bias

   bias = Bias(
       model="gpt2",              # Model name (required)
       layer=6,                   # Target steering layer
       api_key=None,              # Neuronpedia API key
       device="auto",             # Device for model
       dtype="float16",           # Model precision
       library_path="concepts.json"  # Concept library file
   )

Model Parameter
~~~~~~~~~~~~~~~

The ``model`` parameter accepts:

**Shorthand names** (automatically configured):

- ``"gpt2"`` → GPT-2 Small
- ``"gpt2-medium"`` → GPT-2 Medium
- ``"gpt2-large"`` → GPT-2 Large
- ``"gpt2-xl"`` → GPT-2 XL

**HuggingFace model IDs**:

.. code-block:: python

   bias = Bias("meta-llama/Llama-2-7b-hf")
   bias = Bias("mistralai/Mistral-7B-v0.1")

Layer Parameter
~~~~~~~~~~~~~~~

Which transformer layer to apply steering to:

.. code-block:: python

   # Early layers: syntactic features
   bias = Bias("gpt2", layer=2)

   # Middle layers: semantic features (recommended)
   bias = Bias("gpt2", layer=6)

   # Later layers: output-focused features
   bias = Bias("gpt2", layer=10)

Default layers by model:

.. list-table::
   :header-rows: 1

   * - Model
     - Total Layers
     - Default
   * - gpt2
     - 12
     - 6
   * - gpt2-medium
     - 24
     - 12
   * - gpt2-large
     - 36
     - 18
   * - gpt2-xl
     - 48
     - 24

Device Parameter
~~~~~~~~~~~~~~~~

Where to load the model:

- ``"auto"`` - Automatic selection (GPU if available)
- ``"cuda"`` - NVIDIA GPU
- ``"cuda:0"`` - Specific GPU
- ``"cpu"`` - CPU only
- ``"mps"`` - Apple Silicon GPU

.. code-block:: python

   # Automatic (recommended)
   bias = Bias("gpt2", device="auto")

   # Force CPU
   bias = Bias("gpt2", device="cpu")

   # Specific GPU
   bias = Bias("gpt2", device="cuda:1")

Dtype Parameter
~~~~~~~~~~~~~~~

Model precision:

- ``"float16"`` - Half precision (default, fastest)
- ``"bfloat16"`` - Brain float (better for some GPUs)
- ``"float32"`` - Full precision (slowest, most accurate)

.. code-block:: python

   # Fast inference
   bias = Bias("gpt2", dtype="float16")

   # Maximum accuracy
   bias = Bias("gpt2", dtype="float32")

NeuronpediaConfig
-----------------

For low-level control, use ``NeuronpediaConfig``:

.. code-block:: python

   from bias.core import NeuronpediaConfig

   config = NeuronpediaConfig(
       api_key="your-api-key",           # API key for rate limits
       base_url="https://neuronpedia.org/api",  # API base URL
       model_id="gpt2-small",            # Neuronpedia model ID
       layer=6,                          # SAE layer
       sae_id="res-jb"                   # SAE variant
   )

SAE Variants
~~~~~~~~~~~~

The ``sae_id`` parameter selects which SAE to use:

- ``"res-jb"`` - Residual stream SAE (recommended)
- Others may be available per model

ModelConfig
-----------

For advanced model loading:

.. code-block:: python

   from bias.core import ModelConfig

   config = ModelConfig(
       model_name="gpt2",
       device="auto",
       dtype="float16",
       load_in_8bit=False,       # 8-bit quantization
       load_in_4bit=False,       # 4-bit quantization
       trust_remote_code=False   # For custom model code
   )

Quantization
~~~~~~~~~~~~

For large models, use quantization:

.. code-block:: python

   from bias.core import SteeringEngine, NeuronpediaConfig, ModelConfig

   model_config = ModelConfig(
       model_name="meta-llama/Llama-2-7b-hf",
       load_in_8bit=True  # Reduces memory by ~50%
   )

   np_config = NeuronpediaConfig(
       model_id="llama-7b",
       layer=16
   )

   engine = SteeringEngine(
       "meta-llama/Llama-2-7b-hf",
       neuronpedia_config=np_config,
       model_config=model_config
   )

SteeringConfig
--------------

Configure steering behavior:

.. code-block:: python

   from bias.core.config import SteeringConfig

   config = SteeringConfig(
       default_intensity=1.0,    # Default steering strength
       num_features=5,           # Features per concept
       auto_balance=True,        # Weight by relevance
       normalize_vectors=True    # Normalize steering vectors
   )

Environment Variables
---------------------

Set defaults via environment variables:

.. code-block:: bash

   # Neuronpedia API key
   export NEURONPEDIA_API_KEY="your-key"

   # Default model
   export BIAS_DEFAULT_MODEL="gpt2-medium"

   # Default device
   export BIAS_DEVICE="cuda"

   # Concept library path
   export BIAS_LIBRARY_PATH="/path/to/library.json"

Configuration Files
-------------------

Create a ``bias.json`` file for project defaults:

.. code-block:: json

   {
     "model": "gpt2-medium",
     "layer": 12,
     "device": "auto",
     "dtype": "float16",
     "library_path": "project_concepts.json",
     "neuronpedia": {
       "model_id": "gpt2-medium",
       "sae_id": "res-jb"
     }
   }

Load with:

.. code-block:: python

   import json
   from bias import Bias

   with open("bias.json") as f:
       config = json.load(f)

   bias = Bias(**config)

Best Practices
--------------

1. **Start with defaults** - They work well for most cases

2. **Use appropriate model size**:
   
   - Prototyping: ``gpt2``
   - Production: ``gpt2-medium`` or larger

3. **Use GPU when available**:

   .. code-block:: python

      bias = Bias("gpt2", device="cuda")

4. **Cache your API key**:

   .. code-block:: bash

      export NEURONPEDIA_API_KEY="your-key"

5. **Save successful concepts**:

   .. code-block:: python

      bias.steer("professional", save=True)

