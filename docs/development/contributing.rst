Contributing Guide
==================

We welcome contributions to Bias! This guide will help you get started.

Setting Up Development Environment
----------------------------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/codewithdark-git/bias.git
      cd bias

2. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # Linux/macOS
      # or
      venv\Scripts\activate  # Windows

3. **Install development dependencies**:

   .. code-block:: bash

      pip install -e ".[dev]"

4. **Install pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

Code Style
----------

We use these tools for code quality:

- **Black** for formatting
- **Ruff** for linting
- **mypy** for type checking

Run them manually:

.. code-block:: bash

   # Format
   black bias/

   # Lint
   ruff check bias/

   # Type check
   mypy bias/

Running Tests
-------------

We use pytest for testing:

.. code-block:: bash

   # Run all tests
   pytest

   # With coverage
   pytest --cov=bias

   # Specific test file
   pytest tests/test_engine.py

   # Verbose output
   pytest -v

Pull Request Process
--------------------

1. **Fork the repository** on GitHub

2. **Create a feature branch**:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

3. **Make your changes**

4. **Run tests and linting**:

   .. code-block:: bash

      pytest
      ruff check bias/
      black --check bias/

5. **Commit with descriptive message**:

   .. code-block:: bash

      git commit -m "Add feature: description of what it does"

6. **Push and create PR**:

   .. code-block:: bash

      git push origin feature/your-feature-name

7. **Fill out the PR template** on GitHub

What to Contribute
------------------

We're especially interested in:

- **Bug fixes** - Found a bug? Please fix it!
- **Documentation** - Improvements, examples, typo fixes
- **Tests** - Increase test coverage
- **New model support** - Add Neuronpedia mappings for new models
- **Performance** - Make things faster
- **Examples** - Real-world use cases

Design Principles
-----------------

When contributing, please follow these principles:

1. **Simplicity first** - The high-level API should be simple
2. **Composability** - Features should work together
3. **Interpretability** - Users should understand what's happening
4. **Documentation** - Every public function needs docstrings
5. **Testing** - New features need tests

Documentation
-------------

Build docs locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

Then open ``_build/html/index.html``.

Docstring format (NumPy style):

.. code-block:: python

   def my_function(param1: str, param2: int = 10) -> bool:
       """
       Short description.

       Longer description if needed.

       Parameters
       ----------
       param1 : str
           Description of param1
       param2 : int, default=10
           Description of param2

       Returns
       -------
       bool
           Description of return value

       Examples
       --------
       >>> my_function("hello", 5)
       True
       """

Reporting Issues
----------------

When reporting bugs, please include:

- Python version
- Bias version
- Operating system
- Minimal reproduction code
- Full error traceback

Feature requests should include:

- Use case description
- Proposed API
- Expected behavior

Code of Conduct
---------------

Be respectful and constructive. We're all here to make AI more interpretable!

Questions?
----------

- Open a GitHub issue
- Email: codewithdark90@gmail.com

