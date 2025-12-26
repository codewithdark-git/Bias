ConceptLibrary API Reference
============================

Library for storing and retrieving concept-feature mappings.

ConceptLibrary Class
--------------------

.. autoclass:: bias.core.library.ConceptLibrary
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from bias.core import ConceptLibrary

   library = ConceptLibrary("my_concepts.json")

   # Add a concept
   library.add_concept(
       concept="professional",
       feature_ids=[1234, 5678, 9012],
       model_id="gpt2-small",
       layer=6,
       notes="Optimized for business emails"
   )

   # Retrieve it later
   feature_ids = library.get_concept(
       "professional",
       model_id="gpt2-small",
       layer=6
   )

Listing Concepts
~~~~~~~~~~~~~~~~

.. code-block:: python

   # All concepts
   concepts = library.list_concepts()

   # Filter by model
   concepts = library.list_concepts(model_id="gpt2-small")

   # Search
   matches = library.search_concepts("formal")

Managing Concepts
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Update
   library.update_concept(
       "professional",
       model_id="gpt2-small",
       layer=6,
       notes="Updated notes"
   )

   # Remove
   library.remove_concept(
       "professional",
       model_id="gpt2-small",
       layer=6
   )

Import/Export
~~~~~~~~~~~~~

.. code-block:: python

   # Export to dictionary
   data = library.export_to_dict()

   # Import from dictionary
   library.import_from_dict(data, overwrite=True)

   # Share between projects
   import json
   with open("shared_concepts.json", "w") as f:
       json.dump(library.export_to_dict(), f)

