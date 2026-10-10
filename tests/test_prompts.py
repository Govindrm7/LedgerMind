import hashlib

from ledgermind.data.prompts import (
    FUNCTION_DOCS,
    INSTRUCTIONS,
    INSTRUCTIONS_V2,
    build_messages,
    build_prompt,
)
from ledgermind.document import Document

DOC = Document.build("T", [["", "2020"], ["revenue", "$ 5829"]], ["revenue grew ."])


def test_prompt_contains_document_question_and_cue():
    prompt = build_prompt("what was revenue?", DOC)
    assert prompt.startswith(INSTRUCTIONS)
    assert "r1 | revenue | $ 5829" in prompt
    assert "[s0] revenue grew ." in prompt
    assert prompt.endswith("### Question\nwhat was revenue?\n\n### JSON\n")


def test_instructions_list_the_executor_vocabulary():
    for token in ("pct_change", "mean", "1000000", "100", "abstain"):
        assert token in INSTRUCTIONS


def test_messages_carry_the_identical_prompt():
    (msg,) = build_messages("q", DOC)
    assert msg == {"role": "user", "content": build_prompt("q", DOC)}


def test_v1_text_is_frozen():
    # The released SFT and GRPO models were trained on this exact text.
    assert hashlib.sha256(INSTRUCTIONS.encode()).hexdigest() == (
        "fe0964b0199dc80d3ac032124ed2b928ee4bdf5645d690357773ae265a1f091d"
    )


def test_default_prompt_is_the_training_version():
    assert build_prompt("q", DOC) == build_prompt("q", DOC, "v1")


def test_v2_documents_every_function_and_the_unit_rule():
    from ledgermind.dsl import FUNCTION_NAMES

    assert set(FUNCTION_DOCS) == set(FUNCTION_NAMES)
    prompt = build_prompt("q", DOC, "v2")
    assert prompt.startswith(INSTRUCTIONS_V2)
    assert "pct_change(old, new) = (new - old) / old * 100" in prompt
    assert "Answer in the units the document reports" in prompt
    for name in FUNCTION_NAMES:
        assert f"{name}(" in INSTRUCTIONS_V2
