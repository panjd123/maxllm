import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from typing import Optional
import json
import asyncio

from .maxllm import get_completer, async_openai_complete

app = typer.Typer(help="MaxLLM CLI - Unified OpenAI API client with rate limiting and caching")
console = Console()


@app.command()
def sleep(model: str = typer.Argument(..., help="Model name to put to sleep")):
    """Put a local model to sleep."""
    try:
        completer = get_completer(model)
        completer._vllm_sleep()
        console.print(f"[green]✓[/green] Model '{model}' has been put to sleep")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def is_sleep(model: str = typer.Argument(..., help="Model name to check")):
    """Check if a local model is sleeping."""
    try:
        completer = get_completer(model)
        sleeping = completer._vllm_is_sleep()
        if sleeping:
            console.print(f"[yellow]Model '{model}' is sleeping[/yellow]")
        else:
            console.print(f"[green]Model '{model}' is awake[/green]")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def wakeup(model: str = typer.Argument(..., help="Model name to wake up")):
    """Wake up a local model."""
    try:
        completer = get_completer(model)
        completer._vllm_wake_up()
        console.print(f"[green]✓[/green] Model '{model}' has been woken up")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)

@app.command()
def test(model: str = typer.Argument(..., help="Model name to test")):
    """Test if a local model is responsive."""
    try:
        response = asyncio.run(async_openai_complete(model=model, prompt="Say hello!"))
        console.print(f"[green]✓[/green] Model '{model}' is responsive. Sample response: {response}")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)

@app.command()
def chat(
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model name to use"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="User prompt"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    json_mode: bool = typer.Option(False, "--json", help="Enable JSON mode"),
    raw: bool = typer.Option(False, "--raw", help="Return raw response object"),
):
    """Get chat completion from a model."""
    try:
        # If no prompt provided via option, read from stdin or prompt user
        if not prompt:
            if not typer.get_text_stream('stdin').isatty():
                # Read from stdin if piped
                import sys
                prompt = sys.stdin.read().strip()
            else:
                # Interactive mode
                prompt = typer.prompt("Enter your prompt")

        console.print(f"\n[cyan]Sending request to model: {model}[/cyan]")

        response = asyncio.run(async_openai_complete(
            model=model,
            prompt=prompt,
            system_prompt=system,
            json_mode=json_mode,
            raw=raw,
        ))

        console.print("\n[green]Response:[/green]")
        if json_mode or isinstance(response, dict):
            # Pretty print JSON
            console.print(Panel(
                Syntax(json.dumps(response, indent=2, ensure_ascii=False), "json", theme="monokai"),
                title="JSON Response",
                border_style="green"
            ))
        else:
            # Print text response
            console.print(Panel(response, border_style="green"))

    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()