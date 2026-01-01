Configuration Guide
===================

This guide covers all configuration options in Bias.

BiasConfig (Recommended)
------------------------

The ``BiasConfig`` class is the recommended way to configure Bias. It provides
a unified interface for all settings and supports environment variables.

.. code-block:: python

   from bias import Bias, BiasConfig

   config = BiasConfig(
       # API Settings
       api_key="your-neuronpedia-api-key",
       
       # Model Settings
       model="gpt2",
       layer=6,
       sae_id="res-jb",
       
       # Device Settings
       device="auto",
       dtype="float16",
       
       # Steering Defaults
       intensity=1.0,
       num_features=5,
       
       # Library Settings
       library_path="bias_concepts.json",
   )

   bias = Bias(config=config)

BiasConfig Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``api_key``
     - str
     - None
     - Neuronpedia API key for higher rate limits
   * - ``model``
     - str
     - "gpt2"
     - Model name (shorthand or HuggingFace ID)
   * - ``layer``
     - int
     - auto
     - Transformer layer for steering
   * - ``sae_id``
     - str
     - "res-jb"
     - SAE variant identifier
   * - ``device``
     - str
     - "auto"
     - Device for model ("auto", "cuda", "cpu", "mps")
   * - ``dtype``
     - str
     - "float16"
     - Model precision
   * - ``intensity``
     - float
     - 1.0
     - Default steering intensity
   * - ``num_features``
     - int
     - 5
     - Default features per concept
   * - ``library_path``
     - str
     - "bias_concepts.json"
     - Path for concept library

API Key Configuration
---------------------

Your Neuronpedia API key unlocks higher rate limits for feature searches.
There are three ways to provide it:

**1. Explicit Parameter:**

.. code-block:: python

   config = BiasConfig(api_key="your-api-key")

**2. Environment Variable (Recommended):**

.. code-block:: bash

   export NEURONPEDIA_API_KEY="your-api-key"

.. code-block:: python

   # API key automatically loaded from environment
   config = BiasConfig(model="gpt2")

**3. Using from_env():**

.. code-block:: python

   config = BiasConfig.from_env()

Environment Variables
---------------------

BiasConfig supports these environment variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
     - Example
   * - ``NEURONPEDIA_API_KEY``
     - API key for Neuronpedia
     - ``your-api-key``
   * - ``BIAS_MODEL``
     - Default model name
     - ``gpt2-medium``
   * - ``BIAS_LAYER``
     - Default layer number
     - ``12``
   * - ``BIAS_DEVICE``
     - Default device
     - ``cuda``

Set them in your shell:

.. code-block:: bash

   export NEURONPEDIA_API_KEY="your-key"
   export BIAS_MODEL="gpt2-medium"
   export BIAS_DEVICE="cuda"

Then use ``from_env()``:

.. code-block:: python

   from bias import BiasConfig, Bias

   config = BiasConfig.from_env()
   bias = Bias(config=config)

Quick Configure Function
------------------------

For convenience, use the ``configure()`` helper:

.. code-block:: python

   from bias import configure, Bias

   config = configure(
       api_key="your-key",
       model="gpt2-medium",
       device="cuda",
   )
   bias = Bias(config=config)

Model Parameter
---------------

The ``model`` parameter accepts:

**Shorthand names** (automatically configured):

- ``"gpt2"`` → GPT-2 Small (layer 6)
- ``"gpt2-medium"`` → GPT-2 Medium (layer 12)
- ``"gpt2-large"`` → GPT-2 Large (layer 18)
- ``"gpt2-xl"`` → GPT-2 XL (layer 24)

**HuggingFace model IDs**:

.. code-block:: python

   config = BiasConfig(model="meta-llama/Llama-2-7b-hf")

Layer Parameter
---------------

Which transformer layer to apply steering to:

.. code-block:: python

   # Early layers: syntactic features
   config = BiasConfig(model="gpt2", layer=2)

   # Middle layers: semantic features (recommended)
   config = BiasConfig(model="gpt2", layer=6)

   # Later layers: output-focused features
   config = BiasConfig(model="gpt2", layer=10)

Default layers by model:

.. list-table::
   :header-rows: 1

   * - Model
     - Total Layers
     - Default Layer
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
----------------

Where to load the model:

- ``"auto"`` - Automatic selection (GPU if available)
- ``"cuda"`` - NVIDIA GPU
- ``"cuda:0"`` - Specific GPU
- ``"cpu"`` - CPU only
- ``"mps"`` - Apple Silicon GPU

.. code-block:: python

   # Automatic (recommended)
   config = BiasConfig(device="auto")

   # Force CPU
   config = BiasConfig(device="cpu")

   # Specific GPU
   config = BiasConfig(device="cuda:1")

Dtype Parameter
---------------

Model precision:

- ``"float16"`` - Half precision (default, fastest)
- ``"bfloat16"`` - Brain float (better for some GPUs)
- ``"float32"`` - Full precision (slowest, most accurate)

.. code-block:: python

   # Fast inference
   config = BiasConfig(dtype="float16")

   # Maximum accuracy
   config = BiasConfig(dtype="float32")

Legacy Configuration
--------------------

The ``Bias`` class also accepts individual parameters directly:

.. code-block:: python

   from bias import Bias

   bias = Bias(
       model="gpt2",
       layer=6,
       api_key="your-key",
       device="auto",
   )

However, using ``BiasConfig`` is recommended for better organization.

NeuronpediaConfig (Advanced)
----------------------------

For low-level control, use ``NeuronpediaConfig``:

.. code-block:: python

   from bias.core import NeuronpediaConfig

   config = NeuronpediaConfig(
       api_key="your-api-key",
       base_url="https://www.neuronpedia.org",
       model_id="gpt2-small",
       layer=6,
       sae_id="res-jb"
   )

ModelConfig (Advanced)
----------------------

For advanced model loading:

.. code-block:: python

   from bias.core import ModelConfig

   config = ModelConfig(
       model_name="gpt2",
       device="auto",
       dtype="float16",
       load_in_8bit=False,
       load_in_4bit=False,
       trust_remote_code=False
   )

Quantization
~~~~~~~~~~~~

For large models, use quantization:

.. code-block:: python

   config = BiasConfig(
       model="meta-llama/Llama-2-7b-hf",
       load_in_8bit=True,  # Reduces memory by ~50%
   )

SteeringConfig (Advanced)
-------------------------

Configure steering behavior:

.. code-block:: python

   from bias.core.config import SteeringConfig

   config = SteeringConfig(
       default_intensity=1.0,
       num_features=5,
       auto_balance=True,
       normalize_vectors=True
   )

Converting Between Configs
--------------------------

``BiasConfig`` can generate internal configs:

.. code-block:: python

   from bias import BiasConfig

   config = BiasConfig(api_key="key", model="gpt2")

   # Get internal configurations
   np_config = config.to_neuronpedia_config()
   model_config = config.to_model_config()
   steering_config = config.to_steering_config()

Serialization
-------------

Save and load configurations:

.. code-block:: python

   import json
   from bias import BiasConfig

   # Save to dict
   config = BiasConfig(model="gpt2-medium", layer=12)
   config_dict = config.to_dict()

   # Save to file
   with open("bias_config.json", "w") as f:
       json.dump(config_dict, f, indent=2)

   # Load from file
   with open("bias_config.json") as f:
       data = json.load(f)
   config = BiasConfig.from_dict(data)

Best Practices
--------------

1. **Use BiasConfig** - It provides the cleanest interface

2. **Store API key in environment**:
   
   .. code-block:: bash

      export NEURONPEDIA_API_KEY="your-key"

3. **Start with defaults** - They work well for most cases

4. **Use appropriate model size**:
   
   - Prototyping: ``gpt2``
   - Production: ``gpt2-medium`` or larger

5. **Use GPU when available**:

   .. code-block:: python

      config = BiasConfig(model="gpt2", device="cuda")

6. **Save successful concepts**:

   .. code-block:: python

      bias.steer("professional", save=True)
