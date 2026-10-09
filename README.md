# LedgerMind

**The LLM proposes. A deterministic verifier decides every number.**

LLMs are unreliable at multistep financial arithmetic. LedgerMind fine tunes a small open model to extract figures with source citations and write a computation plan, never the final number. A deterministic executor runs the plan, and a verifier checks every cited figure against the source document. The system returns an answer with a full audit trail, or refuses with a reason.

```
Question + Document
      │
      ▼
 LLM (SFT + GRPO)  ──►  evidence with source pointers + computation plan
      │
      ▼
 Executor          ──►  the number
      │
      ▼
 Verifier          ──►  provenance, invariants, document integrity
      │
      ▼
 Answer + audit trail, or REJECT with reason
```

## Status

Work in progress. Results, benchmarks and reproduction steps will be published here as each stage lands.

## License

Apache 2.0. See [LICENSE](LICENSE).
