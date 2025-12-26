SteeringEngine API Reference
============================

Low-level engine for full control over steering.

SteeringEngine Class
--------------------

.. autoclass:: bias.core.engine.SteeringEngine
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Engine Usage
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias.core import SteeringEngine, NeuronpediaConfig

   config = NeuronpediaConfig(
       model_id="gpt2-small",
       layer=6,
       sae_id="res-jb"
   )

   engine = SteeringEngine("gpt2", neuronpedia_config=config)

   # Steer with concept
   engine.steer_with_concept("professional", intensity=2.0)

   # Generate
   output = engine.generate("Hello!")

   # Clear steering
   engine.clear_steering()

Custom Steering
~~~~~~~~~~~~~~~

.. code-block:: python

   # Steer with specific features
   engine.steer_with_features([1234, 5678], intensities=[2.0, 1.5])

   # Apply custom vector
   import torch
   vector = torch.randn(768)
   engine.steer_with_vector(vector, intensity=1.5)

Comparing Outputs
~~~~~~~~~~~~~~~~~

.. code-block:: python

   engine.steer_with_concept("formal", intensity=3.0)
   results = engine.compare_outputs("Hello!")

   print("Unsteered:", results['unsteered'])
   print("Steered:", results['steered'])

Feature Exploration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test feature at different intensities
   results = engine.explore_feature(
       feature_id=1234,
       test_prompt="Write a message:",
       intensities=[0.0, 1.0, 2.0, 5.0]
   )

   for intensity, output in results.items():
       print(f"Intensity {intensity}: {output}")

