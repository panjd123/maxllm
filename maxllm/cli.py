import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from typing import Optional
import json
import asyncio
from datasets import load_dataset
from timeit import default_timer as timer
import tiktoken
import multiprocessing as mp

from ._maxllm import get_completer, _get_selector, async_openai_complete, batch_async_tqdm, warmup_models, get_call_status, get_maxllm_config_path, awarmup_models
from .compatibility import compatibility_test

app = typer.Typer(help="MaxLLM CLI - Unified OpenAI API client with rate limiting and caching")
console = Console()


@app.command()
def sleep(model: str = typer.Argument(..., help="Model name to put to sleep")):
    """Put a local model to sleep."""
    try:
        completer = get_completer(model)
        completer.vllm_sleep_mode_manager.sleep()
        console.print(f"[green]✓[/green] Model '{model}' has been put to sleep")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)


@app.command()
def is_sleep(model: str = typer.Argument(..., help="Model name to check")):
    """Check if a local model is sleeping."""
    try:
        completer = get_completer(model)
        sleeping = completer.vllm_sleep_mode_manager.is_sleep()
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
        completer.vllm_sleep_mode_manager.wake_up()
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
def test_embedding(model: str = typer.Argument(..., help="Embedding model name to test")):
    """Test if an embedding model is responsive."""
    try:
        embedding = asyncio.run(async_openai_complete(model=model, prompt="Test embedding"))
        console.print(f"[green]✓[/green] Embedding model '{model}' is responsive. Sample embedding length: {len(embedding)}")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)

@app.command()
def ccompatibility(model: str = typer.Argument(..., help="Model name to test compatibility")):
    """Run compatibility tests on a model."""
    try:
        console.print(f"[cyan]Running compatibility tests for model '{model}'...[/cyan]")
        asyncio.run(compatibility_test(model))
        console.print(f"[green]✓[/green] Compatibility tests completed for model '{model}'")
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}")
        raise typer.Exit(1)

def prepare_benchmark_prompts(num_prompt: int, max_tokens: int):
    subset = "narrativeqa"
    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    def truncate_to_tokens(text, max_tokens):
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        ids = ids[:max_tokens]
        return enc.decode(ids)

    ds = load_dataset("THUDM/LongBench", subset, split="test", revision="refs/pr/7")

    messages_list = []

    for i in range(num_prompt):
        row = ds[i]
        context = row["context"]
        question = row["input"]

        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        prompt = truncate_to_tokens(prompt, max_tokens)

        messages = [
            {"role": "user", "content": prompt}
        ]
        messages_list.append(messages)
        
    return messages_list

def benchmark_single_model(model: str, messages_list: list[dict], max_output_tokens: int, pbar_postion: int = 0):
    warmup_models([model])
    tasks = []
    for messages in messages_list:
        tasks.append(async_openai_complete(model=model, messages=messages, max_tokens=max_output_tokens, force=True))
    start_time = timer()
    responses = asyncio.run(batch_async_tqdm(tasks, desc=f"Benchmarking {model}", position=pbar_postion))
    end_time = timer()
    call_status = get_call_status()
    seconds_taken = end_time - start_time
    qps = len(messages_list) / seconds_taken
    return seconds_taken, qps, call_status

def benchmark_multi_models(models: list[str], num_prompt: int, max_tokens: int, max_output_tokens: int):
    messages_list = prepare_benchmark_prompts(num_prompt, max_tokens)
    tune_model_name = []
    with mp.Pool(processes=len(models)) as pool:
        results = []
        for i, model in enumerate(models):
            results.append(pool.apply_async(benchmark_single_model, args=(model, messages_list, max_output_tokens, i)))
        pool.close()
        pool.join()
        for model, result in zip(models, results):
            seconds_taken, qps, call_status = result.get()
            console.print(f"[green]Model: {model} | Time: {seconds_taken:.2f}s | QPS: {qps:.2f}[/green]")
            tune_model_name.append(f"{model}:{qps:.2f}")
    console.print(f"[cyan]Recommended tuned model string: {'+'.join(tune_model_name)}[/cyan]")
    

# maxllm benchmark Qwen3-4B-A100    54.74s  3.65qps
# maxllm benchmark Qwen3-4B-4090    160.59s  1.25qps
# maxllm benchmark Qwen3-4B-3090    206.50s  0.97qps
# maxllm benchmark Qwen3-4B-A100+Qwen3-4B-4090+Qwen3-4B-3090  37.95s  5.27qps
@app.command()
def benchmark(model: str = typer.Argument(..., help="Model name to benchmark"), 
              num_prompt: int = typer.Option(200, "--num-prompt", "-n", help="Number of prompts to generate"),
              max_tokens: int = typer.Option(4096, "--max-tokens", "-t", help="Maximum tokens for input"),
              max_output_tokens: int = typer.Option(512, "--max-output-tokens", "-o", help="Maximum tokens for output")
              ):
    console.print(f"[cyan]Preparing benchmark prompt for model '{model}'...[/cyan]")
    messages_list = prepare_benchmark_prompts(num_prompt, max_tokens)
    console.print(f"[cyan]Benchmarking model '{model}' with {num_prompt} prompts...[/cyan]")
    warmup_models([model])
    seconds_taken, qps, call_status = benchmark_single_model(model, messages_list, max_output_tokens)
    console.print(f"[green]Call status: {call_status}[/green]")
    console.print(f"[green]Benchmark completed in {seconds_taken:.2f} seconds[/green]")
    console.print(f"[green]Average requests per second: {qps:.2f}[/green]")

# maxllm tune llama-A100+llama-A6000+llama-4090+llama-3090 --num-prompt 200 --max-tokens 2420 --max-output-tokens 50
@app.command()
def tune(model: str = typer.Argument(..., help="Model name to benchmark"), 
              num_prompt: int = typer.Option(200, "--num-prompt", "-n", help="Number of prompts to generate"),
              max_tokens: int = typer.Option(4096, "--max-tokens", "-t", help="Maximum tokens for input"),
              max_output_tokens: int = typer.Option(512, "--max-output-tokens", "-o", help="Maximum tokens for output")
              ):
    models = _get_selector(model).models
    console.print(f"[cyan]Tuning and benchmarking models: {models} with {num_prompt} prompts...[/cyan]")
    benchmark_multi_models(models, num_prompt, max_tokens, max_output_tokens)
    

@app.command()
def edit(editor: str = typer.Argument("code", help="Editor to use")):
    """Open the editor for configuration."""
    import os
    import subprocess
    config_path = get_maxllm_config_path()
    if not os.path.exists(config_path):
        console.print(f"[red]✗[/red] Configuration file does not exist at {config_path}")
        raise typer.Exit(1)
    try:
        subprocess.run([editor, config_path])
    except Exception as e:
        console.print(f"[red]✗[/red] Error opening editor: {e}")
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
