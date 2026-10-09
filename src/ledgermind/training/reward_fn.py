"""Adapter from the LedgerMind reward to TRL's GRPO reward function interface.

TRL calls ``reward_fn(prompts=..., completions=..., **dataset_columns)`` with one entry
per sampled completion. We compute the verifier based reward once per completion and
push each reward term to TRL's metric logger, so training curves show
``reward/correct``, ``reward/fabrication`` and the rest side by side. A hack shows up as
one term moving while ``correct`` does not.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache

from ledgermind.rewards import DEFAULT_WEIGHTS, RewardWeights, compute_reward
from ledgermind.training.records import document_from_json
from ledgermind.verifier.integrity import check_integrity


@lru_cache(maxsize=4096)
def _integrity(payload: str):
    return check_integrity(document_from_json(payload))


def make_reward_fn(
    weights: RewardWeights = DEFAULT_WEIGHTS, match_mode: str = "scale"
) -> Callable[..., list[float]]:
    def ledgermind_reward(
        prompts: list,
        completions: list,
        document_json: list[str],
        gold_json: list[str],
        log_metric: Callable[[str, float], None] | None = None,
        **_: object,
    ) -> list[float]:
        totals: list[float] = []
        sums: dict[str, float] = defaultdict(float)
        statuses: dict[str, int] = defaultdict(int)
        for completion, doc_payload, gold_payload in zip(
            completions, document_json, gold_json, strict=True
        ):
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            result = compute_reward(
                text,
                document_from_json(doc_payload),
                json.loads(gold_payload),
                weights=weights,
                match_mode=match_mode,
                integrity_issues=_integrity(doc_payload),
            )
            totals.append(result.total)
            for term, value in result.terms.items():
                sums[term] += value
            statuses[result.verdict.status if result.verdict else "invalid"] += 1
        if log_metric is not None and totals:
            n = len(totals)
            for term, value in sums.items():
                log_metric(f"reward/{term}", value / n)
            for status in ("accepted", "rejected", "abstained", "invalid"):
                log_metric(f"verdict/{status}", statuses[status] / n)
        return totals

    return ledgermind_reward
