Quick Start Guide
=================

This guide will get you up and running with Bias in just a few minutes.

Basic Usage
-----------

The simplest way to use Bias is through the high-level ``Bias`` class:

.. code-block:: python

   from bias import Bias

   # Initialize with a model
   bias = Bias("gpt2")

   # Steer toward a concept
   bias.steer("professional writing")

   # Generate text
   output = bias.generate("Write an email about:")
   print(output)

   # Reset steering when done
   bias.reset()

Using Configuration (Recommended)
---------------------------------

For production use, create a ``BiasConfig`` with your API key:

.. code-block:: python

   from bias import Bias, BiasConfig

   # Create configuration with your Neuronpedia API key
   config = BiasConfig(
       api_key="your-neuronpedia-api-key",
       model="gpt2",
       device="auto",
   )

   # Pass config to Bias
   bias = Bias(config=config)

   # Now steer and generate
   bias.steer("professional formal writing", intensity=2.0)
   output = bias.generate("Write an email about the project:")
   print(output)

Using Environment Variables
---------------------------

Set your API key as an environment variable for convenience:

.. code-block:: bash

   export NEURONPEDIA_API_KEY="your-api-key"

Then create config without explicitly passing the key:

.. code-block:: python

   from bias import Bias, BiasConfig

   # API key automatically loaded from environment
   config = BiasConfig(model="gpt2")
   bias = Bias(config=config)

   # Or use from_env() for full environment configuration
   config = BiasConfig.from_env(model="gpt2-medium")
   bias = Bias(config=config)

Quick Configure Function
------------------------

Use the ``configure()`` helper for one-liner configuration:

.. code-block:: python

   from bias import configure, Bias

   config = configure(api_key="your-key", model="gpt2-medium")
   bias = Bias(config=config)

Method Chaining
---------------

Bias supports fluent method chaining for concise code:

.. code-block:: python

   from bias import Bias

   output = (
       Bias("gpt2")
       .steer("creative poetic", intensity=2.0)
       .generate("The sunset painted the sky with")
   )
   print(output)

Context Manager
---------------

Use Bias as a context manager for automatic cleanup:

.. code-block:: python

   with Bias("gpt2") as bias:
       bias.steer("technical documentation")
       output = bias.generate("How to implement:")
   # Steering automatically cleared

Comparing Outputs
-----------------

See the effect of steering by comparing outputs:

.. code-block:: python

   bias = Bias("gpt2")
   bias.steer("extremely formal academic", intensity=3.0)

   results = bias.compare("Explain photosynthesis:")
   print("Without steering:", results['unsteered'])
   print("With steering:", results['steered'])

Functional API
--------------

For quick scripting, use the functional API:

.. code-block:: python

   from bias import steer, generate, reset_steering

   steer("professional formal writing")
   output = generate("Dear colleague,")
   print(output)

   reset_steering()

Command Line Interface
----------------------

Bias includes a powerful CLI:

.. code-block:: bash

   # Generate with steering
   bias generate "Write a poem:" -c "romantic" -i 2.0

   # Discover features
   bias discover "technical language"

   # Interactive mode
   bias interactive

Next Steps
----------

- Learn about :doc:`concepts` to understand how steering works
- Explore :doc:`../user_guide/configuration` for complete configuration options
- Check the :doc:`../api/bias` for complete API documentation
