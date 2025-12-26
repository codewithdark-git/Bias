Core Concepts
=============

This page explains the key concepts behind Bias and LLM steering.

What is LLM Steering?
---------------------

**Steering** is a technique for controlling language model behavior by
modifying the model's internal representations during generation.

Unlike prompt engineering (which works from outside the model) or fine-tuning
(which changes model weights), steering works by:

1. Identifying directions in the model's activation space that correspond to specific behaviors
2. Adding these directions to activations during forward passes
3. The model's outputs shift toward the corresponding behaviors

Think of it like gently nudging the model's "thought process" in a particular direction.

Sparse Autoencoders (SAEs)
--------------------------

**Sparse Autoencoders** are neural networks trained to decompose model activations
into interpretable features.

.. image:: ../_static/sae_diagram.png
   :alt: SAE Architecture
   :width: 600px

Key properties:

- **Sparsity**: Only a few features activate at once
- **Interpretability**: Each feature often corresponds to a human-understandable concept
- **Decomposition**: Complex activations are broken into simpler components

Each SAE feature has:

- A **description**: What concept the feature represents
- A **weight vector**: The direction in activation space
- **Examples**: Texts that strongly activate this feature

Neuronpedia
-----------

`Neuronpedia <https://neuronpedia.org>`_ is a public database of SAE features
for various language models.

Bias connects to Neuronpedia to:

- **Search** for features matching concept descriptions
- **Retrieve** feature weight vectors for steering
- **Explore** feature activations and examples

Steering Intensity
------------------

The **intensity** parameter controls how strongly the steering affects the model:

.. list-table::
   :header-rows: 1

   * - Intensity Range
     - Effect
   * - 0.5 - 1.0
     - Subtle influence
   * - 1.0 - 2.0
     - Moderate effect
   * - 2.0 - 5.0
     - Strong effect
   * - 5.0+
     - Very strong (may reduce coherence)

Higher intensity means stronger steering, but too high can cause:

- Repetitive outputs
- Loss of coherence
- Degraded text quality

Start low (1.0) and increase gradually.

Feature Combination
-------------------

Bias can combine multiple features for nuanced steering:

.. code-block:: python

   # Multiple concepts combined
   bias.steer("professional")
   bias.steer("friendly")
   bias.steer("concise")

Or using specific feature IDs:

.. code-block:: python

   bias.steer_features([1234, 5678, 9012], intensity=1.5)

Features are combined by adding their weight vectors, allowing
complex behavioral modifications.

Layers
------

Steering can target different **layers** in the transformer:

- **Early layers (0-4)**: Token-level, syntactic features
- **Middle layers (5-8)**: Semantic, conceptual features
- **Later layers (9+)**: Task-specific, output-focused features

Bias uses a sensible default layer for each model, but you can override:

.. code-block:: python

   bias = Bias("gpt2", layer=8)  # Use layer 8 instead of default

Best Practices
--------------

1. **Start with low intensity**: Begin at 1.0 and increase gradually
2. **Use specific concepts**: "formal business email" works better than "good writing"
3. **Combine complementary features**: "professional" + "concise" work well together
4. **Test and iterate**: Use ``compare()`` to see the effect of steering
5. **Clear steering between tasks**: Use ``reset()`` to avoid interference

