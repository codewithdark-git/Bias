Bias API Reference
==================

High-level API for LLM steering.

Bias Class
----------

.. autoclass:: bias.api.Bias
   :members:
   :undoc-members:
   :show-inheritance:

Functional API
--------------

For quick scripting without creating a Bias instance.

.. autofunction:: bias.api.steer

.. autofunction:: bias.api.generate

.. autofunction:: bias.api.discover_features

.. autofunction:: bias.api.reset_steering

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from bias import Bias

   # Initialize
   bias = Bias("gpt2")

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

