What is LLM Steering?
=====================

Large Language Models (LLMs) are powerful but can be difficult to control.
**Steering** is an emerging technique that provides precise, interpretable
control over model behavior.

The Control Problem
-------------------

Traditional methods for controlling LLM behavior have significant limitations:

**Prompt Engineering**

- Fragile: Small changes can cause large behavior shifts
- Verbose: Requires long, detailed instructions
- Inconsistent: Same prompt may give different results
- Limited: Can't access model internals

**Fine-tuning**

- Expensive: Requires significant compute
- Data-hungry: Needs curated training data
- Irreversible: Hard to undo without retraining
- Risky: Can cause catastrophic forgetting

**RLHF (Reinforcement Learning from Human Feedback)**

- Complex: Requires reward modeling
- Slow: Iterative training process
- Mode collapse: Can reduce output diversity
- Alignment tax: May reduce capabilities

The Steering Approach
---------------------

Steering takes a fundamentally different approach. Instead of changing the
model or its inputs, steering modifies the model's internal computations
during inference.

**Key Insight**: Neural networks encode information in specific directions
in their activation spaces. By identifying and manipulating these directions,
we can control model behavior.

How Steering Works
------------------

1. **Identify Behavior Directions**
   
   Researchers find vectors in activation space that correspond to specific
   behaviors (e.g., formality, sentiment, technical language).

2. **Create Steering Vectors**
   
   These directions are extracted as steering vectors - numerical
   representations of behavioral concepts.

3. **Apply During Generation**
   
   During text generation, steering vectors are added to the model's
   activations. This "nudges" the model toward the corresponding behavior.

4. **Control Intensity**
   
   The magnitude of the steering vector controls how strongly the
   behavior is induced.

.. code-block:: python

   # Conceptual example
   activations = model.get_activations(input_tokens)
   steered_activations = activations + intensity * steering_vector
   output = model.generate_from(steered_activations)

Advantages of Steering
----------------------

**Interpretable**
   Each steering vector corresponds to a specific, understandable concept.

**Reversible**
   Steering can be turned on/off instantly without modifying the model.

**Precise**
   Multiple behaviors can be combined and controlled independently.

**Efficient**
   No retraining required - works at inference time.

**Composable**
   Different steering vectors can be combined for nuanced control.

Steering with SAE Features
--------------------------

The most interpretable form of steering uses **Sparse Autoencoder (SAE) features**.
SAEs decompose model activations into interpretable components, each representing
a specific concept or pattern.

Bias leverages Neuronpedia's database of labeled SAE features to provide:

- Natural language search for steering concepts
- Pre-computed steering vectors
- Interpretable feature descriptions
- Example activations for understanding

This makes steering accessible without deep technical knowledge of
neural network internals.

Research Background
-------------------

Steering builds on several research directions:

- **Activation Patching**: Identifying causal effects of activations
- **Representation Engineering**: Controlling model behavior via representations
- **Mechanistic Interpretability**: Understanding model internals
- **Sparse Autoencoders**: Decomposing activations into features

See :doc:`research` for relevant papers and resources.

