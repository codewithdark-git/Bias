Advanced Usage Guide
====================

This guide covers advanced features for power users.

Using Specific Feature IDs
--------------------------

For precise control, use specific Neuronpedia feature IDs:

.. code-block:: python

   from bias import Bias

   bias = Bias("gpt2")

   # Steer with known feature IDs
   bias.steer_features([1234, 5678, 9012], intensity=2.0)

   output = bias.generate("Hello!")

Different intensities per feature:

.. code-block:: python

   bias.steer_features(
       [1234, 5678, 9012],
       intensity=[2.0, 1.5, 1.0]  # Per-feature intensities
   )

Discovering Features
--------------------

Explore available features for a concept:

.. code-block:: python

   features = bias.discover("sarcasm", num_features=10)

   for f in features:
       print(f"#{f['id']}: {f['description']}")
       print(f"   Score: {f['score']:.3f}")
       print(f"   Examples: {f.get('activating_examples', [])[:2]}")

Exploring Feature Behavior
--------------------------

Test a feature at different intensities:

.. code-block:: python

   results = bias.explore(
       feature_id=1234,
       test_prompt="Hello, how are you?"
   )

   for intensity, output in results.items():
       print(f"Intensity {intensity}:")
       print(f"  {output}")

Concept Library
---------------

Save and reuse concept-feature mappings:

.. code-block:: python

   # Save a concept
   features = bias.discover("professional", num_features=5)
   bias.save_concept(
       name="my-professional-style",
       feature_ids=[f['id'] for f in features],
       notes="Optimized for emails"
   )

   # Load later (no API call needed)
   feature_ids = bias.load_concept("my-professional-style")
   if feature_ids:
       bias.steer_features(feature_ids, intensity=2.0)

   # List all saved concepts
   print(bias.list_saved_concepts())

Low-Level Engine Access
-----------------------

For full control, use the ``SteeringEngine`` directly:

.. code-block:: python

   from bias.core import SteeringEngine, NeuronpediaConfig

   config = NeuronpediaConfig(
       api_key="your-api-key",  # Optional
       model_id="gpt2-small",
       layer=6,
       sae_id="res-jb"
   )

   engine = SteeringEngine(
       "gpt2",
       neuronpedia_config=config
   )

   # Full control over steering
   engine.steer_with_concept(
       "formal writing",
       intensity=2.5,
       num_features=10,
       auto_balance=True,  # Weight by relevance scores
       layer=8  # Override default layer
   )

   output = engine.generate(
       "Hello",
       max_new_tokens=100,
       temperature=0.8,
       top_p=0.95
   )

   engine.clear_steering()

Custom Configuration
--------------------

Full configuration options:

.. code-block:: python

   from bias import Bias

   bias = Bias(
       model="gpt2-medium",
       layer=12,                    # Target layer
       api_key="your-key",          # Neuronpedia API key
       device="cuda",               # "auto", "cuda", "cpu", "mps"
       dtype="float16",             # "float16", "bfloat16", "float32"
       library_path="concepts.json" # Custom library location
   )

For custom models:

.. code-block:: python

   bias = Bias(
       model="your-org/your-model",
       neuronpedia_id="matching-id",  # Neuronpedia model ID
       sae_id="res-jb",               # SAE variant
       layer=16                        # Target layer
   )

Direct Neuronpedia Access
-------------------------

Access Neuronpedia API directly:

.. code-block:: python

   from bias.core import NeuronpediaClient, NeuronpediaConfig

   config = NeuronpediaConfig(
       model_id="gpt2-small",
       layer=6,
       sae_id="res-jb"
   )

   client = NeuronpediaClient(config)

   # Search features
   features = client.search_features("technical jargon", top_k=20)

   # Get feature details
   details = client.get_feature_details(1234)
   print(details['description'])
   print(details['activating_examples'])

   # Get steering vector directly
   vector = client.get_feature_vector(1234)
   print(vector.shape)

   # Combine multiple features
   combined = client.get_multi_feature_vector(
       [1234, 5678],
       weights=[2.0, 1.0],
       normalize=True
   )

Custom Steering Vectors
-----------------------

Apply your own steering vectors:

.. code-block:: python

   import torch

   # Create or load a custom vector
   custom_vector = torch.randn(768)  # GPT-2 hidden size

   engine.steer_with_vector(
       steering_vector=custom_vector,
       intensity=2.0,
       layer=6
   )

Combining with Other Techniques
-------------------------------

Bias works alongside other generation techniques:

.. code-block:: python

   # With custom generation parameters
   output = bias.generate(
       "Write a story",
       max_tokens=500,
       temperature=0.9,
       top_p=0.95,
       repetition_penalty=1.2,
       do_sample=True
   )

   # With system prompts
   prompt = """You are a helpful assistant.

   User: Write a professional email
   Assistant:"""

   bias.steer("professional")
   output = bias.generate(prompt)

Batch Processing
----------------

For processing multiple prompts:

.. code-block:: python

   prompts = [
       "Write about cats:",
       "Write about dogs:",
       "Write about birds:",
   ]

   bias.steer("scientific", intensity=2.0)

   results = []
   for prompt in prompts:
       output = bias.generate(prompt, max_tokens=100)
       results.append(output)

Performance Tips
----------------

1. **Reuse the Bias instance** - Model loading is expensive

   .. code-block:: python

      # Good - load once
      bias = Bias("gpt2")
      for prompt in prompts:
          output = bias.generate(prompt)

      # Bad - loads model each time
      for prompt in prompts:
          output = Bias("gpt2").generate(prompt)

2. **Cache concept lookups** - Use the library

   .. code-block:: python

      # First time - discovers and saves
      bias.steer("professional", save=True)

      # Later - uses cached features
      bias.steer("professional")

3. **Use appropriate model size** - Smaller is faster

   - gpt2 (117M) - Fastest, good for prototyping
   - gpt2-medium (345M) - Balance of speed and quality
   - gpt2-large+ - Best quality, slower

4. **Use GPU when available**

   .. code-block:: python

      bias = Bias("gpt2", device="cuda")

