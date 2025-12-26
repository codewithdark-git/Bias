"""
Bias CLI
========

Command-line interface for the Bias library.

Examples
--------
$ bias --help
$ bias steer "formal writing" --model gpt2 --intensity 2.0
$ bias generate "Write a poem:" --concept "creative"
$ bias discover "professional language" --num-features 10
$ bias interactive --model gpt2
$ bias explore 1234 --test-prompt "Hello, "
"""

import click
import sys
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

console = Console()


# Lazy loading to avoid slow imports on --help
def get_bias():
    """Lazy import of Bias class."""
    from bias.api import Bias
    return Bias


def get_concept_library():
    """Lazy import of ConceptLibrary."""
    from bias.core.library import ConceptLibrary
    return ConceptLibrary


@click.group()
@click.version_option(version="0.1.0", prog_name="bias")
def app():
    """
    Bias - LLM Steering with Interpretable SAE Features
    
    Steer language models toward specific behaviors using
    Neuronpedia's Sparse Autoencoder features.
    
    \b
    Quick Start:
      $ bias interactive              # Start interactive mode
      $ bias discover "formal"        # Find features for a concept
      $ bias generate "Hello" -c "professional"  # Generate with steering
    """
    pass


@app.command()
@click.argument("prompt")
@click.option(
    "-c", "--concept",
    help="Concept to steer toward (e.g., 'professional')",
)
@click.option(
    "-i", "--intensity",
    type=float,
    default=1.0,
    help="Steering intensity (default: 1.0)",
)
@click.option(
    "-m", "--model",
    default="gpt2",
    help="Model to use (default: gpt2)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=100,
    help="Maximum tokens to generate (default: 100)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature (default: 0.7)",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare steered vs unsteered output",
)
def generate(
    prompt: str,
    concept: Optional[str],
    intensity: float,
    model: str,
    max_tokens: int,
    temperature: float,
    compare: bool,
):
    """
    Generate text with optional steering.
    
    \b
    Examples:
      $ bias generate "Write a poem about love:"
      $ bias generate "Dear Sir," -c "formal" -i 2.0
      $ bias generate "Hello" --compare -c "friendly"
    """
    Bias = get_bias()
    
    with console.status("[bold green]Loading model..."):
        bias = Bias(model)
    
    if concept:
        console.print(f"\n[bold cyan]Steering toward:[/] {concept}")
        bias.steer(concept, intensity=intensity)
    
    if compare and concept:
        console.print("\n[bold]Comparing outputs...[/]\n")
        results = bias.compare(prompt, max_tokens)
        
        console.print(Panel(
            results['unsteered'],
            title="[red]Unsteered[/]",
            expand=False
        ))
        console.print(Panel(
            results['steered'],
            title="[green]Steered[/]",
            expand=False
        ))
    else:
        output = bias.generate(prompt, max_tokens, temperature=temperature)
        console.print(f"\n[bold green]Output:[/]\n{output}")


@app.command()
@click.argument("concept")
@click.option(
    "-n", "--num-features",
    type=int,
    default=5,
    help="Number of features to discover (default: 5)",
)
@click.option(
    "-m", "--model",
    default="gpt2",
    help="Model to use (default: gpt2)",
)
@click.option(
    "--save",
    is_flag=True,
    help="Save discovered features to library",
)
def discover(
    concept: str,
    num_features: int,
    model: str,
    save: bool,
):
    """
    Discover SAE features for a concept.
    
    \b
    Examples:
      $ bias discover "formal writing"
      $ bias discover "humor" -n 10
      $ bias discover "professional" --save
    """
    Bias = get_bias()
    
    with console.status("[bold green]Loading model..."):
        bias = Bias(model)
    
    features = bias.discover(concept, num_features)
    
    if not features:
        console.print("[red]No features found[/]")
        return
    
    # Display as table
    table = Table(title=f"Features for '{concept}'")
    table.add_column("ID", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Score", style="yellow")
    
    for f in features:
        table.add_row(
            str(f.get('id', 'N/A')),
            f.get('description', 'No description')[:50],
            f"{f.get('score', 0):.3f}",
        )
    
    console.print(table)
    
    if save and features:
        bias.save_concept(
            name=concept,
            feature_ids=[f['id'] for f in features],
        )
        console.print(f"\n[green]✓ Saved concept '{concept}' to library[/]")


@app.command()
@click.argument("feature_id", type=int)
@click.option(
    "-m", "--model",
    default="gpt2",
    help="Model to use (default: gpt2)",
)
@click.option(
    "--test-prompt",
    default="Write a brief message: ",
    help="Prompt to test with",
)
def explore(
    feature_id: int,
    model: str,
    test_prompt: str,
):
    """
    Explore a feature at different intensities.
    
    \b
    Examples:
      $ bias explore 1234
      $ bias explore 5678 --test-prompt "Hello, "
    """
    Bias = get_bias()
    
    with console.status("[bold green]Loading model..."):
        bias = Bias(model)
    
    results = bias.explore(feature_id, test_prompt)
    
    console.print(f"\n[bold]Feature #{feature_id} at different intensities:[/]\n")
    
    for intensity, output in results.items():
        console.print(f"[cyan]Intensity {intensity}:[/]")
        console.print(f"  {output[:150]}...\n")


@app.command()
@click.option(
    "-m", "--model",
    default="gpt2",
    help="Model to use (default: gpt2)",
)
@click.option(
    "--layer",
    type=int,
    default=None,
    help="Layer to use for steering",
)
def interactive(model: str, layer: Optional[int]):
    """
    Start interactive steering session.
    
    \b
    Commands:
      concept <text>    - Steer with a concept
      features <ids>    - Steer with feature IDs (comma-separated)
      generate <prompt> - Generate text
      compare <prompt>  - Compare steered vs unsteered
      discover <text>   - Discover features for a concept
      clear             - Clear steering
      quit              - Exit
    
    \b
    Examples:
      $ bias interactive
      $ bias interactive --model gpt2-medium --layer 12
    """
    Bias = get_bias()
    
    console.print(Panel.fit(
        "[bold cyan]Bias Interactive Mode[/]\n"
        "Type 'help' for commands, 'quit' to exit",
        border_style="cyan",
    ))
    
    with console.status("[bold green]Loading model..."):
        kwargs = {}
        if layer is not None:
            kwargs['layer'] = layer
        bias = Bias(model, **kwargs)
    
    console.print(f"[green]✓ Model loaded: {model}[/]\n")
    
    help_text = """
[bold]Commands:[/]
  [cyan]concept <text>[/]     - Steer toward a concept
  [cyan]intensity <n>[/]      - Set intensity (default: 1.0)
  [cyan]features <id,id>[/]   - Steer with specific feature IDs
  [cyan]generate <prompt>[/]  - Generate text
  [cyan]compare <prompt>[/]   - Compare steered vs unsteered
  [cyan]discover <text>[/]    - Find features for a concept
  [cyan]clear[/]              - Remove steering
  [cyan]status[/]             - Show current status
  [cyan]help[/]               - Show this help
  [cyan]quit[/]               - Exit
    """
    
    current_intensity = 1.0
    
    while True:
        try:
            command = console.input("\n[bold green]bias>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not command:
            continue
        
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        try:
            if cmd == "quit" or cmd == "exit":
                break
            
            elif cmd == "help":
                console.print(help_text)
            
            elif cmd == "concept":
                if not arg:
                    console.print("[red]Usage: concept <text>[/]")
                    continue
                bias.steer(arg, intensity=current_intensity)
                console.print(f"[green]✓ Steering toward '{arg}'[/]")
            
            elif cmd == "intensity":
                if not arg:
                    console.print(f"[cyan]Current intensity: {current_intensity}[/]")
                else:
                    current_intensity = float(arg)
                    console.print(f"[green]✓ Intensity set to {current_intensity}[/]")
            
            elif cmd == "features":
                if not arg:
                    console.print("[red]Usage: features <id1,id2,...>[/]")
                    continue
                ids = [int(x.strip()) for x in arg.split(",")]
                bias.steer_features(ids, intensity=current_intensity)
                console.print(f"[green]✓ Applied {len(ids)} features[/]")
            
            elif cmd == "generate" or cmd == "gen":
                if not arg:
                    console.print("[red]Usage: generate <prompt>[/]")
                    continue
                output = bias.generate(arg, max_tokens=100)
                console.print(f"\n{output}")
            
            elif cmd == "compare":
                if not arg:
                    console.print("[red]Usage: compare <prompt>[/]")
                    continue
                results = bias.compare(arg)
                console.print(Panel(results['unsteered'], title="[red]Unsteered[/]"))
                console.print(Panel(results['steered'], title="[green]Steered[/]"))
            
            elif cmd == "discover":
                if not arg:
                    console.print("[red]Usage: discover <concept>[/]")
                    continue
                bias.discover(arg)
            
            elif cmd == "clear":
                bias.reset()
                console.print("[green]✓ Steering cleared[/]")
            
            elif cmd == "status":
                status = "active" if bias.is_steering else "inactive"
                console.print(f"[cyan]Steering: {status}[/]")
                console.print(f"[cyan]Intensity: {current_intensity}[/]")
            
            else:
                # Treat as prompt if steering is active
                if bias.is_steering:
                    output = bias.generate(command, max_tokens=100)
                    console.print(f"\n{output}")
                else:
                    console.print(f"[red]Unknown command: {cmd}[/]")
                    console.print("Type 'help' for available commands")
        
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
    
    console.print("\n[cyan]Goodbye![/]")


@app.command()
@click.option(
    "-m", "--model",
    default=None,
    help="Filter by model",
)
def library(model: Optional[str]):
    """
    List saved concepts in the library.
    
    \b
    Examples:
      $ bias library
      $ bias library -m gpt2-small
    """
    ConceptLibrary = get_concept_library()
    lib = ConceptLibrary()
    
    concepts = lib.list_concepts(model_id=model)
    
    if not concepts:
        console.print("[yellow]No saved concepts found[/]")
        return
    
    table = Table(title="Saved Concepts")
    table.add_column("Concept", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Layer", style="yellow")
    table.add_column("Features", style="magenta")
    
    for c in concepts:
        table.add_row(
            c['concept'],
            c['model_id'],
            str(c['layer']),
            str(len(c['feature_ids'])),
        )
    
    console.print(table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

