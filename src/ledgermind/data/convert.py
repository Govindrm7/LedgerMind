"""Convert FinQA gold programs into LedgerMind supervision targets.

A FinQA program such as ``subtract(5829, 5735), divide(#0, 5735)`` becomes::

    evidence: e1 = 5829 at (row 2, col 1), e2 = 5735 at (row 2, col 2)
    plan:     (e1 - e2) / e2

Every numeric argument is located in the document. Supporting facts (``gold_inds``) are
searched first, then the whole document; table cells win over text on ties. Two FinQA
conventions are made explicit in the plan rather than hidden in the evidence:

* a percentage argument ``9%`` means 0.09, so it becomes ``(e1 / 100)`` with e1 = 9
* table operations over percent cells (``22%``) likewise use ``(e1 / 100)``
* an argument that is the magnitude of a printed negative (the program says 23158, the
  cell says -23158) becomes ``(-e1)`` with e1 = -23158

Examples that cannot be fully grounded are reported with a reason and left out of SFT.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from ledgermind.data.finqa import FinQAExample
from ledgermind.document import Document
from ledgermind.dsl import ALLOWED_LITERALS, compile_plan
from ledgermind.numbers import extract_numbers, numbers_equal, parse_scalar
from ledgermind.schema import REASONING_MAX_CHARS, Answer, Evidence, Source, TableSource, TextSource

_STEP_RE = re.compile(r"(\w+)\((.*?)\)(?:,\s*|$)")
_BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
_TABLE_OPS = {"table_average": "mean", "table_sum": "sum", "table_max": "max", "table_min": "min"}
_PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
_ATOM = 3
_PERCENT_WORDS = re.compile(r"\bpercent(age)?\b|%")
_ECHO_RE = re.compile(r"\(\s*[-+]?\$?\s*[\d.,]*\s*%?\s*\)")
_RATIO_WORDS = re.compile(r"\bratio\b|\bproportion\b|\bportion\b|\bshare\b")


class ConversionError(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


@dataclass(frozen=True)
class ConvertedExample:
    id: str
    question: str
    document: Document
    target: Answer
    gold_answer: float | bool
    flags: tuple[str, ...] = ()


@dataclass
class ConversionReport:
    total: int = 0
    converted: int = 0
    failures: Counter[str] = field(default_factory=Counter)
    flags: Counter[str] = field(default_factory=Counter)

    @property
    def coverage(self) -> float:
        return self.converted / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "converted": self.converted,
            "coverage": round(self.coverage, 4),
            "failures": dict(self.failures.most_common()),
            "flags": dict(self.flags.most_common()),
        }


@dataclass(frozen=True)
class _Expr:
    text: str
    prec: int
    words: str


def _gold_indices(ex: FinQAExample, prefix: str, limit: int) -> list[int]:
    indices = (int(k[len(prefix) :]) for k in ex.gold_inds if k.startswith(prefix))
    return [i for i in indices if 0 <= i < limit]


def _wrap(e: _Expr, min_prec: int) -> str:
    return e.text if e.prec >= min_prec else f"({e.text})"


class _Builder:
    """Accumulates evidence for one example while its program is translated."""

    def __init__(self, ex: FinQAExample):
        self.ex = ex
        self.doc = ex.document
        self.evidence: list[Evidence] = []
        self._by_key: dict[tuple, str] = {}
        self.flags: set[str] = set()
        # A few upstream examples carry out of range indices such as text_-1; skip them.
        self.gold_rows = _gold_indices(ex, "table_", self.doc.n_rows)
        self.gold_sents = _gold_indices(ex, "text_", len(self.doc.sentences))

    # evidence

    def _label(self, source: Source, value: float) -> str:
        if isinstance(source, TableSource):
            row = self.doc.row_label(source.row).strip()
            col = self.doc.column_header(source.col).strip()
            return f"{row} ({col})" if col else row
        sentence = self.doc.sentences[source.sent]
        for m in extract_numbers(sentence):
            if numbers_equal(m.value, value):
                before = _ECHO_RE.sub("", sentence[: m.start]).split()[-8:]
                after = _ECHO_RE.sub("", sentence[m.end :]).split()[:3]
                return " ".join([*before, sentence[m.start : m.end], *after])
        return " ".join(_ECHO_RE.sub("", sentence).split()[:12])

    def _add_evidence(self, source: Source, value: float) -> str:
        key = (source.model_dump_json(), value)
        if key not in self._by_key:
            eid = f"e{len(self.evidence) + 1}"
            label = self._label(source, value)[:200]
            self.evidence.append(Evidence(id=eid, value=value, label=label, source=source))
            self._by_key[key] = eid
        return self._by_key[key]

    def _candidates(self) -> list[Source]:
        gold: list[Source] = [
            TableSource(row=r, col=c) for r in self.gold_rows for c in range(len(self.doc.table[r]))
        ]
        gold += [TextSource(sent=s) for s in self.gold_sents]
        rest = [loc for loc in self.doc.locations() if loc not in gold]
        return gold + rest

    def _locate(self, value: float) -> tuple[Source, float, bool]:
        """Find where ``value`` is printed. Returns (source, printed value, sign flipped)."""
        signed = [s for s in self._candidates() if self._printed(s, value)]
        if signed:
            if len(signed) > 1:
                self.flags.add("ambiguous_location")
            return signed[0], value, False
        flipped = [s for s in self._candidates() if self._printed(s, -value)]
        if flipped and value != 0:
            self.flags.add("sign_flip")
            return flipped[0], -value, True
        raise ConversionError("unresolved_number", str(value))

    def _printed(self, source: Source, value: float) -> bool:
        return any(numbers_equal(m.value, value) for m in self.doc.numbers_at(source))

    # program translation

    def number(self, arg: str) -> _Expr:
        parsed = parse_scalar(arg)
        if parsed is None:
            raise ConversionError("bad_argument", arg)
        source, printed, flipped = self._locate(parsed.value)
        eid = self._add_evidence(source, printed)
        if parsed.is_percent:
            self.flags.add("percent_argument")
            text = f"(-{eid} / 100)" if flipped else f"({eid} / 100)"
            return _Expr(text, _ATOM, f"{eid} as a fraction")
        if flipped:
            return _Expr(f"(-{eid})", _ATOM, f"the magnitude of {eid}")
        return _Expr(eid, _ATOM, eid)

    def constant(self, arg: str) -> _Expr:
        raw = arg[len("const_") :]
        value = -1.0 if raw == "m1" else float(raw)
        if abs(value) not in ALLOWED_LITERALS:
            raise ConversionError("literal_not_allowed", arg)
        text = "(-1)" if value < 0 else str(int(value)) if value.is_integer() else str(value)
        return _Expr(text, _ATOM, text.strip("()"))

    def table_row(self, label: str, op: str) -> _Expr:
        target = label.strip().lower()
        rows = [
            r for r in range(self.doc.n_rows) if self.doc.row_label(r).strip().lower() == target
        ]
        if not rows:
            raise ConversionError("unresolved_row", label)
        preferred = [r for r in rows if r in self.gold_rows]
        row = (preferred or rows)[0]
        refs = []
        for c in range(1, len(self.doc.table[row])):
            source = TableSource(row=row, col=c)
            mentions = self.doc.numbers_at(source)
            if mentions:
                eid = self._add_evidence(source, mentions[0].value)
                if mentions[0].is_percent:
                    self.flags.add("percent_argument")
                    eid = f"({eid} / 100)"
                refs.append(eid)
        if not refs:
            raise ConversionError("empty_row", label)
        verb = {"mean": "average", "sum": "sum", "max": "maximum", "min": "minimum"}[_TABLE_OPS[op]]
        words = f"take the {verb} of the '{label}' row"
        return _Expr(f"{_TABLE_OPS[op]}({', '.join(refs)})", _ATOM, words)


def _split_steps(program: str) -> list[tuple[str, list[str]]]:
    steps = []
    pos = 0
    for m in _STEP_RE.finditer(program.strip()):
        if m.start() != pos:
            raise ConversionError("bad_program", program)
        pos = m.end()
        op, inner = m.group(1), m.group(2)
        if op in _TABLE_OPS:
            label = re.sub(r",\s*none\s*$", "", inner)
            steps.append((op, [label]))
        else:
            steps.append((op, [a.strip() for a in inner.split(",")]))
    if pos != len(program.strip()) or not steps:
        raise ConversionError("bad_program", program)
    return steps


def _describe(op: str, args: list[_Expr]) -> str:
    a = args[0].words
    b = args[1].words if len(args) > 1 else ""
    return {
        "add": f"add {a} and {b}",
        "subtract": f"subtract {b} from {a}",
        "multiply": f"multiply {a} by {b}",
        "divide": f"divide {a} by {b}",
        "exp": f"raise {a} to the power {b}",
        "greater": f"check whether {a} is greater than {b}",
    }.get(op, a)


def _answer_unit(question: str, steps: list[tuple[str, list[str]]]) -> str:
    op, args = steps[-1]
    if op == "greater":
        return "boolean"
    if op == "multiply" and "const_100" in args:
        return "percent"
    q = question.lower()
    if op == "divide" and (_PERCENT_WORDS.search(q) or _RATIO_WORDS.search(q)):
        return "ratio"
    return "number"


def convert(ex: FinQAExample) -> ConvertedExample:
    """Translate one example. Raises :class:`ConversionError` if it cannot be grounded."""
    builder = _Builder(ex)
    steps = _split_steps(ex.program)
    results: list[_Expr] = []
    descriptions: list[str] = []

    for i, (op, raw_args) in enumerate(steps):
        if op in _TABLE_OPS:
            expr = builder.table_row(raw_args[0], op)
            results.append(expr)
            descriptions.append(f"Step {i + 1}: {expr.words}.")
            continue
        if len(raw_args) != 2:
            raise ConversionError("bad_arity", f"{op}({', '.join(raw_args)})")
        args: list[_Expr] = []
        for arg in raw_args:
            if arg.startswith("#"):
                k = int(arg[1:])
                if k >= len(results):
                    raise ConversionError("bad_reference", arg)
                prior = results[k]
                args.append(_Expr(prior.text, prior.prec, f"result {k + 1}"))
            elif arg.startswith("const_"):
                args.append(builder.constant(arg))
            else:
                args.append(builder.number(arg))

        if op in _BINARY:
            sym = _BINARY[op]
            text = f"{_wrap(args[0], _PREC[sym])} {sym} {_wrap(args[1], _PREC[sym] + 1)}"
            expr = _Expr(text, _PREC[sym], f"result {i + 1}")
        elif op == "exp":
            expr = _Expr(f"pow({args[0].text}, {args[1].text})", _ATOM, f"result {i + 1}")
        elif op == "greater":
            if i != len(steps) - 1:
                raise ConversionError("nested_boolean", ex.program)
            expr = _Expr(f"greater({args[0].text}, {args[1].text})", _ATOM, f"result {i + 1}")
        else:
            raise ConversionError("unknown_op", op)
        results.append(expr)
        descriptions.append(f"Step {i + 1}: {_describe(op, args)}.")

    plan = results[-1].text
    if not builder.evidence:
        raise ConversionError("no_evidence", ex.program)
    compiled = compile_plan(plan)
    if compiled.refs != {e.id for e in builder.evidence}:
        raise ConversionError("unused_evidence", plan)

    reasoning = " ".join(descriptions)
    if len(reasoning) > REASONING_MAX_CHARS:
        reasoning = reasoning[: REASONING_MAX_CHARS - 3].rstrip() + "..."
    try:
        target = Answer(
            reasoning=reasoning,
            evidence=builder.evidence,
            plan=plan,
            answer_unit=_answer_unit(ex.question, steps),
        )
    except ValidationError as err:
        raise ConversionError("schema_violation", str(err.errors()[0]["msg"])) from err
    flags = tuple(sorted(builder.flags))
    return ConvertedExample(ex.id, ex.question, ex.document, target, ex.gold_answer, flags)


def convert_all(examples: list[FinQAExample]) -> tuple[list[ConvertedExample], ConversionReport]:
    report = ConversionReport()
    converted: list[ConvertedExample] = []
    for ex in examples:
        report.total += 1
        try:
            result = convert(ex)
        except ConversionError as err:
            report.failures[err.reason] += 1
            continue
        converted.append(result)
        report.converted += 1
        report.flags.update(result.flags)
    return converted, report
