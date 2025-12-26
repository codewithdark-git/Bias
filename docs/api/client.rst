NeuronpediaClient API Reference
================================

Client for interacting with Neuronpedia API.

NeuronpediaClient Class
-----------------------

.. autoclass:: bias.core.client.NeuronpediaClient
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Searching Features
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from bias.core import NeuronpediaClient, NeuronpediaConfig

   config = NeuronpediaConfig(
       model_id="gpt2-small",
       layer=6,
       sae_id="res-jb"
   )

   client = NeuronpediaClient(config)

   # Search for features
   features = client.search_features("formal language", top_k=10)

   for f in features:
       print(f"#{f['id']}: {f['description']}")

Getting Feature Details
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   details = client.get_feature_details(1234)

   print(f"Description: {details['description']}")
   print(f"Examples: {details['activating_examples']}")

Getting Steering Vectors
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Single feature vector
   vector = client.get_feature_vector(1234)
   print(f"Vector shape: {vector.shape}")

   # Combined vector from multiple features
   combined = client.get_multi_feature_vector(
       feature_ids=[1234, 5678, 9012],
       weights=[2.0, 1.0, 1.5],
       normalize=True
   )

Feature Discovery
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Discover with explanations
   features = client.discover_features(
       concept="technical language",
       num_features=5,
       verbose=True
   )

