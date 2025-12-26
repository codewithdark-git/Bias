Neuronpedia Integration
=======================

`Neuronpedia <https://neuronpedia.org>`_ is a public database of interpretable
SAE features for various language models. Bias integrates with Neuronpedia
to provide easy access to steering vectors.

What is Neuronpedia?
--------------------

Neuronpedia hosts:

- **Pre-trained SAEs** for popular language models
- **Feature labels** with human-readable descriptions
- **Activation examples** showing where features activate
- **Search functionality** to find features by concept
- **Visualization tools** for exploring features

Supported Models
----------------

Neuronpedia provides SAE features for:

.. list-table::
   :header-rows: 1

   * - Model
     - Neuronpedia ID
     - Available Layers
   * - GPT-2 Small
     - gpt2-small
     - 0-11
   * - GPT-2 Medium
     - gpt2-medium
     - 0-23
   * - GPT-2 Large
     - gpt2-large
     - 0-35
   * - GPT-2 XL
     - gpt2-xl
     - 0-47

Additional models are being added regularly.

API Endpoints
-------------

Bias uses these Neuronpedia API endpoints:

**Search Features**

.. code-block:: text

   GET /api/search
   Parameters:
     - model_id: Model identifier (e.g., "gpt2-small")
     - layer: Layer number
     - sae_id: SAE variant (e.g., "res-jb")
     - query: Search query
     - top_k: Number of results

**Get Feature Details**

.. code-block:: text

   GET /api/feature/{model_id}/{layer}/{sae_id}/{feature_id}
   Returns:
     - description: Human-readable description
     - activating_examples: Texts where feature activates
     - weight_vector: Decoder weights for steering
     - activation_stats: Statistics about activations

Using Neuronpedia in Bias
-------------------------

Bias handles Neuronpedia integration automatically:

.. code-block:: python

   from bias import Bias

   # Bias searches Neuronpedia for "professional" features
   bias = Bias("gpt2")
   bias.steer("professional writing")  # Uses Neuronpedia API

For direct access to the client:

.. code-block:: python

   from bias.core import NeuronpediaClient, NeuronpediaConfig

   config = NeuronpediaConfig(
       model_id="gpt2-small",
       layer=6,
       sae_id="res-jb"
   )

   client = NeuronpediaClient(config)

   # Search for features
   features = client.search_features("formal language")
   print(features)

   # Get feature details
   details = client.get_feature_details(1234)
   print(details['description'])

   # Get steering vector
   vector = client.get_feature_vector(1234)

API Rate Limits
---------------

Neuronpedia has rate limits on API requests. For higher limits:

1. Create a Neuronpedia account
2. Get an API key from settings
3. Configure Bias with the key:

.. code-block:: python

   bias = Bias("gpt2", api_key="your-api-key")

Or via environment variable:

.. code-block:: bash

   export NEURONPEDIA_API_KEY="your-api-key"

Caching
-------

Bias caches feature vectors to reduce API calls. The ``ConceptLibrary``
class stores successful concept-feature mappings:

.. code-block:: python

   # Save a concept for reuse
   bias.save_concept("my-professional-style", feature_ids=[1234, 5678])

   # Load later without API call
   features = bias.load_concept("my-professional-style")

Feature Quality
---------------

Not all Neuronpedia features are equally useful for steering:

**Good features for steering:**

- Clear, specific descriptions
- Consistent activation patterns
- High activation values on relevant texts
- Low activation on unrelated texts

**Features to avoid:**

- Vague or unclear descriptions
- Polysemantic (activate for multiple unrelated things)
- Very rare (dead features)
- Very common (activate for everything)

Bias's search algorithm prioritizes high-quality features, but you can
also explore and select features manually.

Contributing to Neuronpedia
---------------------------

You can contribute to Neuronpedia by:

- Labeling features with descriptions
- Reporting mislabeled features
- Suggesting improvements

Visit `Neuronpedia <https://neuronpedia.org>`_ to explore and contribute.

