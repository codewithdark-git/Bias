Research Background
===================

Bias builds on cutting-edge research in mechanistic interpretability
and representation engineering. This page provides pointers to the
underlying research.

Key Papers
----------

Sparse Autoencoders
~~~~~~~~~~~~~~~~~~~

**Towards Monosemanticity** (Anthropic, 2023)
   Introduces SAEs for extracting interpretable features from language models.
   
   - `Paper <https://transformer-circuits.pub/2023/monosemantic-features/index.html>`_

**Scaling Monosemanticity** (Anthropic, 2024)
   Scales SAE interpretability to Claude 3 Sonnet, finding millions of features.
   
   - `Paper <https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>`_

Activation Steering
~~~~~~~~~~~~~~~~~~~

**Activation Addition** (Turner et al., 2023)
   Demonstrates steering via adding vectors to residual stream.
   
   - `Paper <https://arxiv.org/abs/2308.10248>`_

**Representation Engineering** (Zou et al., 2023)
   Introduces representation reading and control for safety.
   
   - `Paper <https://arxiv.org/abs/2310.01405>`_

**Steering Llama 2 via Contrastive Activation Addition** (Rimsky et al., 2023)
   Shows steering for sycophancy, hallucination, and other behaviors.
   
   - `Paper <https://arxiv.org/abs/2312.06681>`_

Mechanistic Interpretability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021)
   Foundational work on understanding transformer internals.
   
   - `Paper <https://transformer-circuits.pub/2021/framework/index.html>`_

**In-context Learning and Induction Heads** (Olsson et al., 2022)
   Identifies specific circuits responsible for in-context learning.
   
   - `Paper <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>`_

Key Concepts
------------

Residual Stream
~~~~~~~~~~~~~~~

The residual stream is the main "information highway" in transformers.
Each layer reads from and writes to this stream. Steering works by
adding vectors to the residual stream.

.. code-block:: text

   Input Embeddings
        ↓
   [Layer 0] ←→ Residual Stream ← Steering vectors added here
        ↓
   [Layer 1] ←→ Residual Stream
        ↓
      ...
        ↓
   Output Logits

Superposition
~~~~~~~~~~~~~

Language models represent more features than they have dimensions by
using superposition - overlapping, compressed representations. SAEs
decompose this superposition into interpretable features.

Causal Intervention
~~~~~~~~~~~~~~~~~~~

Steering is a form of causal intervention - we modify internal states
to see how they affect outputs. This connects to activation patching
and circuit analysis techniques.

Open Source Tools
-----------------

**TransformerLens**
   Library for mechanistic interpretability research.
   
   - `GitHub <https://github.com/neelnanda-io/TransformerLens>`_

**SAELens**
   Tools for training and analyzing SAEs.
   
   - `GitHub <https://github.com/jbloomAus/SAELens>`_

**Neuronpedia**
   Database of SAE features with visualization and search.
   
   - `Website <https://neuronpedia.org>`_

**Bias** (this library!)
   High-level API for SAE-based steering.

Research Groups
---------------

- `Anthropic <https://www.anthropic.com/research>`_
- `EleutherAI <https://www.eleuther.ai/>`_
- `Conjecture <https://www.conjecture.dev/>`_
- `Apollo Research <https://www.apolloresearch.ai/>`_

Community Resources
-------------------

- `AI Alignment Forum <https://www.alignmentforum.org/>`_
- `LessWrong <https://www.lesswrong.com/>`_
- `Mechanistic Interpretability Discord <https://discord.gg/mechinterp>`_

Future Directions
-----------------

Active research areas include:

- **Scaling SAEs** to larger models
- **Multi-layer steering** for complex behaviors
- **Automatic feature discovery** without human labeling
- **Safety applications** like jailbreak prevention
- **Truthfulness and honesty** steering
- **Combining with RLHF** for better alignment

Bias aims to make these research advances accessible to practitioners
and enable new applications of interpretable AI control.

