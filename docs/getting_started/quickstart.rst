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
- Explore :doc:`../user_guide/advanced_usage` for more features
- Check the :doc:`../api/bias` for complete API documentation

