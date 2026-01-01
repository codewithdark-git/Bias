Bias API Reference
==================

High-level API for LLM steering.

BiasConfig Class
----------------

The recommended way to configure Bias.

.. autoclass:: bias.core.config.BiasConfig
   :members:
   :undoc-members:
   :show-inheritance:

Bias Class
----------

.. autoclass:: bias.api.Bias
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Function
----------------------

.. autofunction:: bias.core.config.configure

Functional API
--------------

For quick scripting without creating a Bias instance.

.. autofunction:: bias.api.steer

.. autofunction:: bias.api.generate

.. autofunction:: bias.api.discover_features

.. autofunction:: bias.api.reset_steering

Usage Examples
--------------

With BiasConfig (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias import Bias, BiasConfig

   # Create configuration with API key
   config = BiasConfig(
       api_key="your-neuronpedia-api-key",
       model="gpt2",
       device="auto",
   )

   # Initialize Bias with config
   bias = Bias(config=config)

   # Steer and generate
   bias.steer("professional writing", intensity=2.0)
   output = bias.generate("Write an email:")
   print(output)

   # Reset
   bias.reset()

Using Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export NEURONPEDIA_API_KEY="your-api-key"

.. code-block:: python

   from bias import Bias, BiasConfig

   # API key loaded from environment
   config = BiasConfig(model="gpt2")
   bias = Bias(config=config)

   # Or use from_env()
   config = BiasConfig.from_env()
   bias = Bias(config=config)

Quick Configure
~~~~~~~~~~~~~~~

.. code-block:: python

   from bias import configure, Bias

   config = configure(api_key="key", model="gpt2-medium")
   bias = Bias(config=config)

Basic Usage (Legacy)
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias import Bias

   # Initialize with parameters directly
   bias = Bias("gpt2", api_key="your-key")

   # Steer and generate
   bias.steer("professional", intensity=2.0)
   output = bias.generate("Write an email:")
   print(output)

   # Reset
   bias.reset()

Method Chaining
~~~~~~~~~~~~~~~

.. code-block:: python

   output = (
       Bias("gpt2")
       .steer("creative")
       .generate("Once upon a time")
   )

Context Manager
~~~~~~~~~~~~~~~

.. code-block:: python

   with Bias("gpt2") as bias:
       bias.steer("formal")
       output = bias.generate("Dear Sir,")
   # Steering automatically cleared

Functional API
~~~~~~~~~~~~~~

.. code-block:: python

   from bias import steer, generate, reset_steering

   steer("professional")
   output = generate("Hello!")
   reset_steering()

Configuration Options
---------------------

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
     - Neuronpedia API key
   * - ``model``
     - str
     - "gpt2"
     - Model name
   * - ``layer``
     - int
     - auto
     - Steering layer
   * - ``device``
     - str
     - "auto"
     - Compute device
   * - ``dtype``
     - str
     - "float16"
     - Model precision
   * - ``intensity``
     - float
     - 1.0
     - Default intensity
   * - ``num_features``
     - int
     - 5
     - Features per concept

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``NEURONPEDIA_API_KEY``
     - API key for Neuronpedia
   * - ``BIAS_MODEL``
     - Default model name
   * - ``BIAS_LAYER``
     - Default layer
   * - ``BIAS_DEVICE``
     - Default device
