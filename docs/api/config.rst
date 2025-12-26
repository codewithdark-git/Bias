Configuration API Reference
===========================

Configuration classes for Bias.

NeuronpediaConfig
-----------------

.. autoclass:: bias.core.config.NeuronpediaConfig
   :members:
   :undoc-members:

ModelConfig
-----------

.. autoclass:: bias.core.config.ModelConfig
   :members:
   :undoc-members:

SteeringConfig
--------------

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

Creating Configurations
~~~~~~~~~~~~~~~~~~~~~~~

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

Serialization
~~~~~~~~~~~~~

.. code-block:: python

   # To dictionary
   data = np_config.to_dict()

   # From dictionary
   new_config = NeuronpediaConfig.from_dict(data)

Getting Model Info
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias.core.config import get_model_config

   info = get_model_config("gpt2")
   print(info['recommended_layer'])  # 6

