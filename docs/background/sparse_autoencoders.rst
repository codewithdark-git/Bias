Sparse Autoencoders (SAEs)
==========================

Sparse Autoencoders are a key technology that makes interpretable
LLM steering possible. This page explains what they are and why they matter.

What is a Sparse Autoencoder?
-----------------------------

A **Sparse Autoencoder** is a neural network trained to:

1. **Encode** model activations into a higher-dimensional, sparse representation
2. **Decode** that representation back to approximate the original activations

The "sparse" part means that only a few dimensions (features) are active
at any given time.

.. code-block:: text

   Model Activations (768 dim)
         ↓
   [Encoder: Linear + ReLU]
         ↓
   Sparse Features (16384 dim, mostly zeros)
         ↓
   [Decoder: Linear]
         ↓
   Reconstructed Activations (768 dim)

Why Sparsity Matters
--------------------

Traditional neural network activations are **dense** - all dimensions
are used simultaneously. This makes interpretation difficult because
each dimension encodes multiple concepts (polysemanticity).

SAEs enforce **sparsity**, which leads to:

- **Monosemantic features**: Each dimension tends to represent one concept
- **Interpretability**: Features can be labeled with human-understandable meanings
- **Decomposability**: Complex activations become sums of simple features

The Superposition Hypothesis
----------------------------

Language models represent more concepts than they have dimensions.
They achieve this through **superposition** - overlapping representations
that share dimensions.

SAEs "unpack" this superposition by:

1. Expanding to a much larger feature space
2. Learning to activate only relevant features
3. Reconstructing the original via combination

This reveals the underlying concepts that were compressed together.

Training SAEs
-------------

SAEs are trained with two objectives:

1. **Reconstruction**: Minimize the difference between input and output
2. **Sparsity**: Penalize having too many active features

.. code-block:: python

   loss = reconstruction_loss + λ * sparsity_penalty

The sparsity penalty encourages the autoencoder to find a minimal set
of features that explain each activation.

SAE Features
------------

Each feature in a trained SAE typically:

- **Activates** for specific concepts, topics, or patterns
- Has a **weight vector** (decoder column) pointing in activation space
- Can be **described** in natural language
- Has **example texts** where it activates strongly

Examples of SAE features:

- "Formal business language"
- "Technical programming terms"
- "Positive emotional sentiment"
- "Questions and inquiries"
- "Geographic locations"

Using SAE Features for Steering
-------------------------------

Each SAE feature has a decoder weight vector. This vector points in
the direction of the corresponding concept in activation space.

To steer toward a concept:

1. Find the feature(s) for that concept
2. Get the decoder weight vector(s)
3. Add the vector(s) to model activations during generation

.. code-block:: python

   # Simplified steering
   feature_vector = sae.decoder[:, feature_id]
   activations = model.layer_output(input)
   steered = activations + intensity * feature_vector
   # Continue forward pass with steered activations

This is what Bias does under the hood, using Neuronpedia's
pre-trained SAE features.

Limitations
-----------

SAE features aren't perfect:

- **Polysemanticity**: Some features still activate for multiple concepts
- **Dead features**: Some features never activate
- **Incomplete coverage**: Not all concepts have corresponding features
- **Model-specific**: SAEs trained on one model don't transfer

Despite these limitations, SAE features provide the most interpretable
approach to LLM steering currently available.

Further Reading
---------------

- `Towards Monosemanticity (Anthropic) <https://transformer-circuits.pub/2023/monosemantic-features/index.html>`_
- `Scaling Monosemanticity (Anthropic) <https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>`_
- `Neuronpedia <https://neuronpedia.org>`_

