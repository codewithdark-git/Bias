Basic Usage Guide
=================

This guide covers the fundamental ways to use Bias for LLM steering.

The Bias Class
--------------

The ``Bias`` class is the main entry point for most users:

.. code-block:: python

   from bias import Bias

   # Initialize with a model name
   bias = Bias("gpt2")

Supported shorthand model names:

- ``gpt2`` - GPT-2 Small (117M parameters)
- ``gpt2-medium`` - GPT-2 Medium (345M)
- ``gpt2-large`` - GPT-2 Large (774M)
- ``gpt2-xl`` - GPT-2 XL (1.5B)

For other models, use the full HuggingFace model ID:

.. code-block:: python

   bias = Bias("meta-llama/Llama-2-7b-hf")

Steering Toward Concepts
------------------------

Use natural language to describe the behavior you want:

.. code-block:: python

   # Steer toward professional writing
   bias.steer("professional formal writing")

   # Steer toward creative expression
   bias.steer("creative imaginative storytelling")

   # Steer toward technical content
   bias.steer("technical programming documentation")

Controlling Intensity
~~~~~~~~~~~~~~~~~~~~~

The ``intensity`` parameter controls steering strength:

.. code-block:: python

   # Subtle effect
   bias.steer("humor", intensity=0.5)

   # Normal effect
   bias.steer("humor", intensity=1.0)

   # Strong effect
   bias.steer("humor", intensity=3.0)

   # Very strong (may affect quality)
   bias.steer("humor", intensity=5.0)

Generating Text
---------------

After steering, generate text with ``generate()``:

.. code-block:: python

   output = bias.generate("Once upon a time")
   print(output)

Customize generation:

.. code-block:: python

   output = bias.generate(
       "Write a story:",
       max_tokens=200,      # Maximum new tokens
       temperature=0.8,      # Higher = more random
   )

Get only the completion (without the prompt):

.. code-block:: python

   completion = bias.complete("Hello, I wanted to")
   print(completion)

Comparing Outputs
-----------------

See the effect of steering by comparing:

.. code-block:: python

   bias.steer("formal academic", intensity=3.0)

   results = bias.compare("Explain gravity:")
   
   print("Without steering:")
   print(results['unsteered'])
   
   print("\nWith steering:")
   print(results['steered'])

Resetting Steering
------------------

Clear all steering to return to baseline:

.. code-block:: python

   bias.reset()

Check if steering is active:

.. code-block:: python

   if bias.is_steering:
       print("Steering is active")

Method Chaining
---------------

Bias methods return ``self`` for fluent chaining:

.. code-block:: python

   output = (
       Bias("gpt2")
       .steer("poetic", intensity=2.0)
       .generate("The wind whispered")
   )

Context Manager
---------------

Use as a context manager for automatic cleanup:

.. code-block:: python

   with Bias("gpt2") as bias:
       bias.steer("technical")
       output = bias.generate("How to implement:")
   # Steering automatically cleared here

Multiple Concepts
-----------------

Apply multiple concepts by calling ``steer()`` multiple times:

.. code-block:: python

   bias.steer("professional", intensity=1.5)
   bias.steer("friendly", intensity=1.0)
   bias.steer("concise", intensity=1.0)

Note: Each ``steer()`` call replaces previous steering.
For combining concepts, see :doc:`advanced_usage`.

Practical Examples
------------------

Email Writing
~~~~~~~~~~~~~

.. code-block:: python

   bias = Bias("gpt2")
   bias.steer("professional business email", intensity=2.0)

   output = bias.generate(
       "Subject: Project Update\n\nDear Team,"
   )

Creative Writing
~~~~~~~~~~~~~~~~

.. code-block:: python

   bias = Bias("gpt2")
   bias.steer("creative fantasy storytelling", intensity=2.5)

   output = bias.generate(
       "In a land where magic flows like rivers,"
   )

Technical Documentation
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   bias = Bias("gpt2")
   bias.steer("clear technical documentation", intensity=2.0)

   output = bias.generate(
       "## Installation\n\nTo install the package,"
   )

Common Issues
-------------

**Output quality decreased**

Try reducing the intensity. Very high intensity (>5.0) can degrade output quality.

**Steering has no effect**

- Check that you're using a supported model
- Try increasing the intensity
- Try different concept descriptions

**Model loading is slow**

The first time you use a model, it downloads from HuggingFace.
Subsequent uses will be faster. Consider using smaller models
for experimentation.

