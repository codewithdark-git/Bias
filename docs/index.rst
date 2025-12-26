.. Bias documentation master file

Bias Documentation
==================

**Bias** is a Python library for steering Large Language Models using
interpretable Sparse Autoencoder (SAE) features from Neuronpedia.

.. image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT

Quick Example
-------------

.. code-block:: python

   from bias import Bias

   # Create a steering instance
   bias = Bias("gpt2")

   # Steer toward professional writing
   bias.steer("professional formal writing")

   # Generate text with the new behavior
   output = bias.generate("Write an email about the project:")
   print(output)  # More formal, professional output!

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/basic_usage
   user_guide/advanced_usage
   user_guide/cli
   user_guide/configuration

.. toctree::
   :maxdepth: 2
   :caption: Background

   background/what_is_steering
   background/sparse_autoencoders
   background/neuronpedia
   background/research

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/bias
   api/engine
   api/client
   api/config
   api/library

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

