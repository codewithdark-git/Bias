# Changelog

All notable changes to Bias will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.1.0] - 2025-12-26

### Initial release! 🎉

### Added

#### High-level API (`Bias` class)

- `steer()` - Steer toward concepts
- `generate()` - Generate text with steering
- `compare()` - Compare steered vs unsteered
- `discover()` - Find features for concepts
- `explore()` - Test features at different intensities

#### Low-level API (`SteeringEngine`)

- Full control over steering parameters
- Custom steering vectors
- Multi-layer steering

#### Neuronpedia Integration (`NeuronpediaClient`)

- Feature search by concept
- Feature details retrieval
- Steering vector extraction

#### Concept Library (`ConceptLibrary`)

- Save and reuse concept-feature mappings
- JSON persistence
- Import/export functionality

#### Command Line Interface

- `bias generate` - Generate with steering
- `bias discover` - Find features
- `bias explore` - Test features
- `bias interactive` - Interactive mode
- `bias library` - Manage saved concepts

#### Documentation

- Getting started guides
- Background on steering and SAEs
- Complete API reference
- Contributing guide

### Supported Models

- GPT-2 (small, medium, large, xl)

---

## Version Guidelines

- **MAJOR** version for incompatible API changes
- **MINOR** version for new features (backwards compatible)
- **PATCH** version for bug fixes
