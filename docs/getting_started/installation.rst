Installation
============

Requirements
------------

Bias requires Python 3.11 or later. It works on Linux, macOS, and Windows.

**Core Dependencies:**

- PyTorch >= 2.5.0
- Transformers >= 4.30.0
- Requests >= 2.28.0
- NumPy >= 1.21.0
- Click >= 8.0.0
- Rich >= 13.0.0

Installing from GitHub
----------------------

Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/codewithdark-git/bias.git

Installing from Source
----------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/codewithdark-git/bias.git
   cd bias
   pip install -e .

Optional Dependencies
---------------------

**Development tools:**

.. code-block:: bash

   pip install "bias[dev] @ git+https://github.com/codewithdark-git/bias.git"

This includes pytest, black, ruff, mypy, and pre-commit.

**Documentation tools:**

.. code-block:: bash

   pip install "bias[docs] @ git+https://github.com/codewithdark-git/bias.git"

This includes Sphinx and related tools.

**Everything:**

.. code-block:: bash

   pip install "bias[all] @ git+https://github.com/codewithdark-git/bias.git"

GPU Support
-----------

For GPU acceleration, ensure you have CUDA-enabled PyTorch installed:

.. code-block:: bash

   # For CUDA 12.4
   pip install torch --index-url https://download.pytorch.org/whl/cu124

Then install Bias:

.. code-block:: bash

   pip install git+https://github.com/codewithdark-git/bias.git

Setting Up Your API Key
-----------------------

Bias uses the Neuronpedia API to search for interpretable features.
While it works without an API key, having one provides higher rate limits.

**Get your API key:**

1. Visit `neuronpedia.org <https://neuronpedia.org>`_
2. Create an account or sign in
3. Navigate to your account settings
4. Copy your API key

**Configure your API key:**

**Option 1: Environment Variable (Recommended)**

Set the environment variable in your shell:

.. code-block:: bash

   # Linux/macOS
   export NEURONPEDIA_API_KEY="your-api-key"

   # Windows (PowerShell)
   $env:NEURONPEDIA_API_KEY="your-api-key"

   # Windows (Command Prompt)
   set NEURONPEDIA_API_KEY=your-api-key

For permanent configuration, add it to your shell profile:

.. code-block:: bash

   # Add to ~/.bashrc or ~/.zshrc
   echo 'export NEURONPEDIA_API_KEY="your-api-key"' >> ~/.bashrc

**Option 2: In Code**

Pass the API key directly to ``BiasConfig``:

.. code-block:: python

   from bias import Bias, BiasConfig

   config = BiasConfig(
       api_key="your-api-key",
       model="gpt2",
   )
   bias = Bias(config=config)

**Option 3: Using from_env()**

Load all configuration from environment:

.. code-block:: python

   from bias import BiasConfig, Bias

   config = BiasConfig.from_env()
   bias = Bias(config=config)

Verifying Installation
----------------------

To verify your installation:

.. code-block:: python

   import bias
   print(f"Bias version: {bias.__version__}")

   # Check if GPU is available
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")

   # Test configuration
   from bias import BiasConfig
   config = BiasConfig(model="gpt2")
   print(f"Config: {config}")

Quick Test
----------

Run a quick test to ensure everything works:

.. code-block:: python

   from bias import Bias, BiasConfig

   # Create config (API key from environment if set)
   config = BiasConfig(model="gpt2")

   # Initialize Bias
   bias = Bias(config=config)

   # Test feature discovery
   features = bias.discover("formal language", num_features=3)
   print(f"Found {len(features)} features")

   # Test generation (optional - requires model download)
   # bias.steer("formal writing")
   # output = bias.generate("Hello,")
   # print(output)

Troubleshooting
---------------

**Installation fails with build errors:**

Make sure you have the latest pip and setuptools:

.. code-block:: bash

   pip install --upgrade pip setuptools wheel

**CUDA not detected:**

Verify PyTorch can see your GPU:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())

**API errors:**

If you see API errors, check your API key:

.. code-block:: python

   from bias import BiasConfig
   config = BiasConfig()
   print(f"API key configured: {config.api_key is not None}")

**Memory errors:**

For large models, use quantization:

.. code-block:: python

   config = BiasConfig(
       model="gpt2-large",
       load_in_8bit=True,
   )
