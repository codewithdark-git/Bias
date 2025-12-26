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

   pip install bias[dev]

This includes pytest, black, ruff, mypy, and pre-commit.

**Documentation tools:**

.. code-block:: bash

   pip install bias[docs]

This includes Sphinx and related tools.

**Everything:**

.. code-block:: bash

   pip install bias[all]

GPU Support
-----------

For GPU acceleration, ensure you have CUDA-enabled PyTorch installed:

.. code-block:: bash

   # For CUDA 12.4
   pip install torch --index-url https://download.pytorch.org/whl/cu124

Then install Bias:

.. code-block:: bash

   pip install git+https://github.com/codewithdark-git/bias.git

Verifying Installation
----------------------

To verify your installation:

.. code-block:: python

   import bias
   print(bias.__version__)

   # Check if GPU is available
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")

