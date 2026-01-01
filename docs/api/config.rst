Configuration API Reference
===========================

Configuration classes for Bias.

BiasConfig (Recommended)
------------------------

The main configuration class for end users.

.. autoclass:: bias.core.config.BiasConfig
   :members:
   :undoc-members:

configure Function
------------------

Helper function for quick configuration.

.. autofunction:: bias.core.config.configure

NeuronpediaConfig
-----------------

Low-level configuration for Neuronpedia API access.

.. autoclass:: bias.core.config.NeuronpediaConfig
   :members:
   :undoc-members:

ModelConfig
-----------

Configuration for model loading.

.. autoclass:: bias.core.config.ModelConfig
   :members:
   :undoc-members:

SteeringConfig
--------------

Configuration for steering behavior.

.. autoclass:: bias.core.config.SteeringConfig
   :members:
   :undoc-members:

Helper Functions
----------------

.. autofunction:: bias.core.config.get_model_config

Constants
---------

SUPPORTED_MODELS
~~~~~~~~~~~~~~~~

Pre-configured model mappings:

.. code-block:: python

   from bias.core.config import SUPPORTED_MODELS

   print(SUPPORTED_MODELS)
   # {
   #     "gpt2": {
   #         "hf_name": "gpt2",
   #         "neuronpedia_id": "gpt2-small",
   #         "recommended_layer": 6,
   #         "sae_id": "res-jb",
   #     },
   #     ...
   # }

Usage Examples
--------------

Using BiasConfig (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias import Bias, BiasConfig

   # Create config with API key
   config = BiasConfig(
       api_key="your-api-key",
       model="gpt2",
       device="auto",
   )

   # Use with Bias
   bias = Bias(config=config)

Using Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export NEURONPEDIA_API_KEY="your-api-key"
   export BIAS_MODEL="gpt2-medium"

.. code-block:: python

   from bias import BiasConfig, Bias

   # Load from environment
   config = BiasConfig.from_env()
   bias = Bias(config=config)

Quick Configure
~~~~~~~~~~~~~~~

.. code-block:: python

   from bias import configure, Bias

   config = configure(api_key="key", model="gpt2")
   bias = Bias(config=config)

Low-Level Configurations
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias.core.config import NeuronpediaConfig, ModelConfig

   # Neuronpedia config
   np_config = NeuronpediaConfig(
       api_key="your-key",
       model_id="gpt2-small",
       layer=6,
       sae_id="res-jb"
   )

   # Model config
   model_config = ModelConfig(
       model_name="gpt2",
       device="cuda",
       dtype="float16"
   )

Converting Configs
~~~~~~~~~~~~~~~~~~

BiasConfig can generate internal configs:

.. code-block:: python

   from bias import BiasConfig

   config = BiasConfig(api_key="key", model="gpt2")

   # Get internal configs
   np_config = config.to_neuronpedia_config()
   model_config = config.to_model_config()
   steering_config = config.to_steering_config()

Serialization
~~~~~~~~~~~~~

.. code-block:: python

   from bias import BiasConfig

   # To dictionary
   config = BiasConfig(model="gpt2")
   data = config.to_dict()

   # From dictionary
   new_config = BiasConfig.from_dict(data)

Getting Model Info
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias.core.config import get_model_config

   info = get_model_config("gpt2")
   print(info['recommended_layer'])  # 6
