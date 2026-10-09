from ledgermind.data.prompts import INSTRUCTIONS, build_messages, build_prompt
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
