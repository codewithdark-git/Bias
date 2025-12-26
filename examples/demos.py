"""
Example Demonstrations
======================

Complete examples showing how to use the Bias library.
"""

from bias import Bias


def basic_example():
    """
    Basic usage example.
    
    This demonstrates the simplest way to use Bias:
    1. Create a Bias instance
    2. Steer toward a concept
    3. Generate text
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Steering")
    print("=" * 60)
    
    # Initialize Bias with GPT-2
    bias = Bias("gpt2")
    
    # Steer toward professional writing
    bias.steer("professional formal writing", intensity=2.0)
    
    # Generate text
    prompt = "Write an email about the quarterly results:"
    output = bias.generate(prompt, max_tokens=100)
    
    print(f"\nPrompt: {prompt}")
    print(f"\nOutput:\n{output}")
    
    # Clean up
    bias.reset()


def multi_concept_example():
    """
    Multi-concept steering example.
    
    This shows how to combine multiple concepts for nuanced steering.
    """
    print("\n" + "=" * 60)
    print("Example 2: Multi-Concept Steering")
    print("=" * 60)
    
    bias = Bias("gpt2")
    
    # Discover features for multiple concepts
    concepts = ["professional", "concise", "friendly"]
    all_feature_ids = []
    
    for concept in concepts:
        print(f"\nDiscovering features for '{concept}'...")
        features = bias.discover(concept, num_features=3)
        all_feature_ids.extend([f['id'] for f in features[:2]])
    
    # Apply combined steering
    print(f"\nApplying {len(all_feature_ids)} combined features...")
    bias.steer_features(all_feature_ids, intensity=1.5)
    
    # Generate
    prompt = "Respond to a customer inquiry about shipping:"
    output = bias.generate(prompt, max_tokens=80)
    
    print(f"\nPrompt: {prompt}")
    print(f"\nOutput:\n{output}")
    
    bias.reset()


def feature_exploration_example():
    """
    Feature exploration example.
    
    This shows how to explore individual features and understand
    their effects at different intensities.
    """
    print("\n" + "=" * 60)
    print("Example 3: Feature Exploration")
    print("=" * 60)
    
    bias = Bias("gpt2")
    
    # Find features for a concept
    print("\nDiscovering features for 'technical language'...")
    features = bias.discover("technical language", num_features=5)
    
    if features:
        # Explore the top feature
        top_feature_id = features[0]['id']
        print(f"\nExploring top feature #{top_feature_id}...")
        results = bias.explore(top_feature_id)
        
        print("\nResults at different intensities:")
        for intensity, output in results.items():
            print(f"\n  Intensity {intensity}:")
            print(f"  {output[:100]}...")
    
    bias.reset()


def concept_library_example():
    """
    Concept library example.
    
    This shows how to save and reuse concept-feature mappings.
    """
    print("\n" + "=" * 60)
    print("Example 4: Using Concept Library")
    print("=" * 60)
    
    bias = Bias("gpt2")
    
    concept = "professional"
    
    # Check if concept exists in library
    cached_features = bias.load_concept(concept)
    
    if cached_features:
        print(f"✓ Using cached features for '{concept}'")
        bias.steer_features(cached_features, intensity=2.0)
    else:
        print(f"Discovering new features for '{concept}'...")
        features = bias.discover(concept, num_features=5)
        feature_ids = [f['id'] for f in features]
        
        # Save to library for future use
        bias.save_concept(
            name=concept,
            feature_ids=feature_ids,
            notes="Optimized for business writing",
        )
        
        bias.steer_features(feature_ids, intensity=2.0)
    
    # Generate
    prompt = "Write a business proposal:"
    output = bias.generate(prompt, max_tokens=100)
    
    print(f"\nPrompt: {prompt}")
    print(f"\nOutput:\n{output}")
    
    # List saved concepts
    print("\n📚 Saved concepts:", bias.list_saved_concepts())
    
    bias.reset()


def comparison_example():
    """
    Comparison example.
    
    This shows how to compare steered vs unsteered outputs.
    """
    print("\n" + "=" * 60)
    print("Example 5: Comparing Outputs")
    print("=" * 60)
    
    bias = Bias("gpt2")
    
    # Steer toward formal writing
    bias.steer("formal academic writing", intensity=2.5)
    
    # Compare outputs
    prompt = "Explain how plants grow:"
    results = bias.compare(prompt, max_tokens=80)
    
    print(f"\nPrompt: {prompt}")
    print(f"\n{'='*40}\nWithout steering:\n{'='*40}")
    print(results['unsteered'])
    print(f"\n{'='*40}\nWith steering:\n{'='*40}")
    print(results['steered'])
    
    bias.reset()


def method_chaining_example():
    """
    Method chaining example.
    
    Bias supports fluent method chaining for concise code.
    """
    print("\n" + "=" * 60)
    print("Example 6: Method Chaining")
    print("=" * 60)
    
    # Method chaining in action
    output = (
        Bias("gpt2")
        .steer("creative poetic", intensity=2.0)
        .generate("The sunset painted the sky with")
    )
    
    print(f"Output:\n{output}")


def context_manager_example():
    """
    Context manager example.
    
    Using Bias as a context manager ensures cleanup.
    """
    print("\n" + "=" * 60)
    print("Example 7: Context Manager")
    print("=" * 60)
    
    with Bias("gpt2") as bias:
        bias.steer("humorous witty", intensity=1.5)
        output = bias.generate("Why did the programmer quit?")
        print(f"Output:\n{output}")
    
    # Steering is automatically cleared when exiting the context


def run_all_examples():
    """Run all examples in sequence."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              Bias Library - Example Demonstrations           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    examples = [
        basic_example,
        multi_concept_example,
        feature_exploration_example,
        concept_library_example,
        comparison_example,
        method_chaining_example,
        context_manager_example,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        print("\n")


if __name__ == "__main__":
    run_all_examples()

