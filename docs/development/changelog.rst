Changelog
=========

All notable changes to Bias will be documented here.

The format is based on `Keep a Changelog <https://keepachangelog.com/>`_.

[Unreleased]
------------

[1.0.0] - 2025-12-26
--------------------

🚀 First Stable Release!

This release introduces a user-friendly configuration system and improved API stability.

Added
~~~~~

- **BiasConfig** - Unified Configuration
  
  - New ``BiasConfig`` class for streamlined configuration
  - Environment variable support (``NEURONPEDIA_API_KEY``, ``BIAS_MODEL``, etc.)
  - ``BiasConfig.from_env()`` for loading config from environment
  - ``configure()`` helper function for quick setup
  - Automatic API key loading from environment

- **Improved API**
  
  - ``Bias`` class now accepts ``config`` parameter
  - Better error handling in Neuronpedia API client
  - Correct API endpoints for Neuronpedia integration

- **Documentation**
  
  - Complete configuration guide
  - API key setup instructions
  - Environment variable documentation
  - Updated examples throughout

Changed
~~~~~~~

- License format updated for newer setuptools compatibility
- Lazy imports to avoid torch dependency during installation

Fixed
~~~~~

- Neuronpedia API endpoint paths
- JSON parsing error handling
- Package installation from GitHub

---

[0.1.0] - 2025-12-26
--------------------

Initial release! 🎉

Added
~~~~~

- **High-level API** (``Bias`` class)
  
  - ``steer()`` - Steer toward concepts
  - ``generate()`` - Generate text with steering
  - ``compare()`` - Compare steered vs unsteered
  - ``discover()`` - Find features for concepts
  - ``explore()`` - Test features at different intensities

- **Low-level API** (``SteeringEngine``)
  
  - Full control over steering parameters
  - Custom steering vectors
  - Multi-layer steering

- **Neuronpedia Integration** (``NeuronpediaClient``)
  
  - Feature search by concept
  - Feature details retrieval
  - Steering vector extraction

- **Concept Library** (``ConceptLibrary``)
  
  - Save and reuse concept-feature mappings
  - JSON persistence
  - Import/export functionality

- **Command Line Interface**
  
  - ``bias generate`` - Generate with steering
  - ``bias discover`` - Find features
  - ``bias explore`` - Test features
  - ``bias interactive`` - Interactive mode
  - ``bias library`` - Manage saved concepts

- **Documentation**
  
  - Getting started guides
  - Background on steering and SAEs
  - Complete API reference
  - Contributing guide

Supported Models
~~~~~~~~~~~~~~~~

- GPT-2 (small, medium, large, xl)

---

Version Guidelines
------------------

- **MAJOR** version for incompatible API changes
- **MINOR** version for new features (backwards compatible)
- **PATCH** version for bug fixes
