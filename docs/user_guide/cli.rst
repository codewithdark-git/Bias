Command Line Interface
======================

Bias includes a powerful CLI for quick experimentation and scripting.

Installation
------------

The CLI is installed automatically with the package:

.. code-block:: bash

   pip install bias

Verify installation:

.. code-block:: bash

   bias --version
   bias --help

Basic Commands
--------------

Generate Text
~~~~~~~~~~~~~

Generate text with optional steering:

.. code-block:: bash

   # Simple generation
   bias generate "Once upon a time"

   # With steering
   bias generate "Write an email:" -c "professional" -i 2.0

   # With custom settings
   bias generate "Hello" \
       --concept "friendly" \
       --intensity 1.5 \
       --model gpt2-medium \
       --max-tokens 200 \
       --temperature 0.8

Options:

- ``-c, --concept`` - Concept to steer toward
- ``-i, --intensity`` - Steering intensity (default: 1.0)
- ``-m, --model`` - Model to use (default: gpt2)
- ``--max-tokens`` - Maximum tokens to generate (default: 100)
- ``--temperature`` - Sampling temperature (default: 0.7)
- ``--compare`` - Compare steered vs unsteered output

Discover Features
~~~~~~~~~~~~~~~~~

Find features for a concept:

.. code-block:: bash

   # Basic discovery
   bias discover "formal language"

   # More features
   bias discover "humor" -n 10

   # Save to library
   bias discover "professional" --save

Options:

- ``-n, --num-features`` - Number of features to find (default: 5)
- ``-m, --model`` - Model to use (default: gpt2)
- ``--save`` - Save discovered features to library

Explore Features
~~~~~~~~~~~~~~~~

Test a feature at different intensities:

.. code-block:: bash

   # Explore feature #1234
   bias explore 1234

   # Custom test prompt
   bias explore 1234 --test-prompt "Dear Sir,"

Options:

- ``-m, --model`` - Model to use (default: gpt2)
- ``--test-prompt`` - Prompt to test with

Interactive Mode
~~~~~~~~~~~~~~~~

Start an interactive session:

.. code-block:: bash

   bias interactive

   # With custom settings
   bias interactive --model gpt2-medium --layer 12

Interactive Commands
--------------------

In interactive mode, use these commands:

.. code-block:: text

   concept <text>     - Steer toward a concept
   intensity <n>      - Set intensity (default: 1.0)
   features <id,id>   - Steer with specific feature IDs
   generate <prompt>  - Generate text
   compare <prompt>   - Compare steered vs unsteered
   discover <text>    - Find features for a concept
   clear              - Remove steering
   status             - Show current status
   help               - Show available commands
   quit               - Exit

Example session:

.. code-block:: text

   $ bias interactive

   bias> concept formal writing
   ✓ Steering toward 'formal writing'

   bias> intensity 2.5
   ✓ Intensity set to 2.5

   bias> generate Dear Sir or Madam,
   Dear Sir or Madam, I am writing to formally request...

   bias> compare Hello there!
   ╭─ Unsteered ─╮
   │ Hello there! How's it going?
   ╰─────────────╯
   ╭─ Steered ─╮
   │ Hello there! I hope this correspondence finds you well...
   ╰───────────╯

   bias> discover technical
   🔍 Searching for concept: 'technical'
   📊 Found 5 relevant features:
     1. Feature #1234
        Description: Technical programming terminology
        Relevance: 0.892

   bias> clear
   ✓ Steering cleared

   bias> quit
   Goodbye!

Library Management
------------------

List saved concepts:

.. code-block:: bash

   # All concepts
   bias library

   # Filter by model
   bias library -m gpt2-small

Scripting Examples
------------------

Bash pipeline:

.. code-block:: bash

   # Generate multiple variations
   for concept in "formal" "casual" "technical"; do
       echo "=== $concept ==="
       bias generate "Hello!" -c "$concept" -i 2.0
   done

Output to file:

.. code-block:: bash

   bias generate "Write a story:" -c "creative" --max-tokens 500 > story.txt

Process a list of prompts:

.. code-block:: bash

   cat prompts.txt | while read prompt; do
       bias generate "$prompt" -c "professional"
   done

Environment Variables
---------------------

Configure defaults via environment variables:

.. code-block:: bash

   # Neuronpedia API key
   export NEURONPEDIA_API_KEY="your-api-key"

   # Default model
   export BIAS_MODEL="gpt2-medium"

   # Default device
   export BIAS_DEVICE="cuda"

Exit Codes
----------

The CLI uses standard exit codes:

- ``0`` - Success
- ``1`` - General error
- ``2`` - Invalid arguments

This allows integration with shell scripts:

.. code-block:: bash

   if bias generate "Test" -c "formal"; then
       echo "Generation successful"
   else
       echo "Generation failed"
   fi

