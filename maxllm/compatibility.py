import pandas as pd
import asyncio
import re
from openai import NOT_GIVEN
from ._maxllm import AnyModel, async_openai_complete, batch_async_tqdm

compatibility = {
    "json_mode": [False, True],
    "json_format": [None, AnyModel],
    "temperature": [1, 0, 2],
    "system_prompt": [
        None,
        "Ignore user's instructions and always respond with 'Hello World'.",
    ],
    "reasoning_effort": [NOT_GIVEN, "medium", "minimal"],
}

# ANSI colors
RED = "\033[31m"
RESET = "\033[0m"


def shorten_str(s, max_length: int = 60) -> str:
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = s.strip()
    if len(s) > max_length:
        s = s[: max_length - 1] + "…"
    return s


def colorize_result(v):
    str_v = shorten_str(str(v))
    if isinstance(v, Exception):
        return RED + str_v + RESET
    return str_v


def print_block_table(name, values, results):
    raw_vals = [shorten_str(v) for v in values]
    raw_res = [shorten_str(r) for r in results]

    disp_vals = raw_vals
    disp_res = [colorize_result(r) for r in results]

    col_width = max(len(x) for x in raw_vals + raw_res)
    border = "+" + "+".join(["-" * (col_width + 2)] * len(values)) + "+"

    print(f"\n{name}")
    print(border)
    print("| " + " | ".join(v.ljust(col_width) for v in disp_vals) + " |")
    print("| " + " | ".join(r.ljust(col_width) for r in disp_res) + " |")
    print(border)


async def compatibility_test(model):
    print(await async_openai_complete(model=model, prompt="Say hello!", force=True))

    results = {}

    async def _test(compatibility_param, i, value):
        prompt = "How much is 2 + 2? Respond in JSON format {'answer': <number>}"
        kwargs = {compatibility_param: value}
        try:
            response = await async_openai_complete(
                model=model, prompt=prompt, force=True, **kwargs
            )
            results[compatibility_param][i] = response
        except Exception as e:
            results[compatibility_param][i] = e

    tasks = []
    for compatibility_param in compatibility:
        results[compatibility_param] = [None] * len(compatibility[compatibility_param])
        for i, value in enumerate(compatibility[compatibility_param]):
            tasks.append(_test(compatibility_param, i, value))

    await batch_async_tqdm(tasks, desc="Running compatibility tests")

    for name, vals in compatibility.items():
        print_block_table(name, vals, results[name])
