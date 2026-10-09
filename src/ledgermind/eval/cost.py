"""Cost per 1,000 queries, self hosted versus API, on the same accounting basis.

Self hosted cost amortizes the GPU hourly price over the measured sustained throughput at
a given concurrency, so it is a cost at full utilization. API cost uses the measured mean
prompt and completion token counts and the provider's list prices. Prices are inputs,
never hardcoded, and every reported figure records the prices it used.
"""

from __future__ import annotations


def self_hosted_cost_per_1k(gpu_usd_per_hour: float, requests_per_second: float) -> float:
    if requests_per_second <= 0:
        raise ValueError("throughput must be positive")
    return gpu_usd_per_hour / 3600.0 / requests_per_second * 1000.0


def api_cost_per_1k(
    mean_prompt_tokens: float,
    mean_completion_tokens: float,
    usd_per_million_input: float,
    usd_per_million_output: float,
) -> float:
    per_query = (
        mean_prompt_tokens * usd_per_million_input + mean_completion_tokens * usd_per_million_output
    ) / 1e6
    return per_query * 1000.0
