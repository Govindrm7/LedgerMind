"""Prompt construction shared by SFT, GRPO, serving and the frontier baselines.

Every system sees byte identical instructions, document rendering and question, so that
accuracy differences come from the model, not the prompt. The base model is trained on
plain text (no chat template); API models receive the same text as a single user turn.
"""

from __future__ import annotations

from ledgermind.document import Document
from ledgermind.dsl import ALLOWED_LITERALS, FUNCTION_NAMES

PROMPT_VERSION = "v1"

_CONSTANTS = ", ".join(str(int(c)) for c in sorted(ALLOWED_LITERALS))
_FUNCTIONS = ", ".join(sorted(FUNCTION_NAMES))

INSTRUCTIONS = f"""You answer questions about a financial document. Do not compute the answer \
yourself. Return one JSON object that cites the figures you need and a plan that computes \
the answer from them. A deterministic executor runs the plan and a verifier checks every \
citation against the document.

JSON format:
{{"reasoning": "<one or two short sentences>",
  "evidence": [{{"id": "e1", "value": <number exactly as printed>, "label": "<what it is>",
                "source": {{"type": "table", "row": <r>, "col": <c>}}}},
               {{"id": "e2", "value": <number>, "label": "<what it is>",
                "source": {{"type": "text", "sent": <s>}}}}],
  "plan": "<expression over e1, e2, ...>",
  "answer_unit": "number" | "percent" | "ratio" | "currency" | "boolean"}}

Rules:
1. Copy each value exactly as printed at the cited location, with its sign. For 12% write \
12, for $ 1,234 write 1234, for ( 45 ) write -45.
2. The plan may use + - * / and parentheses, the functions {_FUNCTIONS}, evidence ids, \
and only these constants: {_CONSTANTS}. Never write the final number in the plan.
3. Convert explicitly in the plan: a printed percentage as a fraction is e1 / 100, \
millions to units is e1 * 1000000.
4. If the document does not contain what the question needs, return \
{{"abstain": true, "reason": "<why>"}}."""


def build_prompt(question: str, doc: Document) -> str:
    return (
        f"{INSTRUCTIONS}\n\n### Document\n{doc.render()}\n\n### Question\n{question}\n\n### JSON\n"
    )


DIRECT_INSTRUCTIONS = """You answer questions about a financial document. Think step by \
step, then give the final answer on its own last line as "Answer: <value>". Write numbers \
without units or thousands separators, a percentage as its number of percent (5.2 for \
5.2%), and yes or no for yes or no questions."""


def build_direct_prompt(question: str, doc: Document) -> str:
    """Baseline prompt: the model computes the answer itself. Same document rendering."""
    return f"{DIRECT_INSTRUCTIONS}\n\n### Document\n{doc.render()}\n\n### Question\n{question}\n"


def build_messages(question: str, doc: Document) -> list[dict[str, str]]:
    """The same prompt as one user message, for chat API baselines."""
    return [{"role": "user", "content": build_prompt(question, doc)}]
